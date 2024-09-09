import pandas as pd
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_csv = pd.read_csv('../test/train_1day.csv')
test_csv = pd.read_csv('../test/testset.csv')

# hyperparameter
w1 = 1  # cost weight
w2 = 0  # peak weight
w3 = 0  # switch weight

class Environment:
    def __init__(self, data):
        self.data = data
        self.battery_cap = 1000
        self.conv_cap = 250
        self.contract = 500
        self.soc_min = 0.1
        self.soc_max = 0.9
        self.total_steps = len(data)
        self.demand_cost = 8320  # 7220, 8320, 9810
        self.cnd_state = [0]

        self.reset()

    def reset(self):
        self.current_step = 0
        self.current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        self.done = False
        self.soc = 0.5
        self.load = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'load'].iloc[0], 2))
        self.pv = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'pv'].iloc[0], 2))
        self.price, self.peak_time = calculate_price(self.current_date)
        self.charge = 0
        self.discharge = 0
        self.ess = self.charge - self.discharge
        self.grid = self.load - self.pv + self.ess
        self.peak = 0
        self.switch = 0
        self.switch_sum = 0
        self.usage_sum = 0
        self.total_cost = (self.peak * self.demand_cost + self.usage_sum) * 0.137
        self.cnd_state = [0]
        self.total_costs = []

        state = [
            float(self.soc.item() if isinstance(self.soc, np.ndarray) else self.soc),
            float(self.load.item() if isinstance(self.load, np.ndarray) else self.load),
            float(self.pv.item() if isinstance(self.pv, np.ndarray) else self.pv),
            float(self.grid.item() if isinstance(self.grid, np.ndarray) else self.grid),
            float(self.total_cost.item() if isinstance(self.total_cost, np.ndarray) else self.total_cost),
            float(self.switch_sum.item() if isinstance(self.switch_sum, np.ndarray) else self.switch_sum),
            float(self.peak.item() if isinstance(self.peak, np.ndarray) else self.peak)
        ]
        return np.array(state)

    def step(self, action):
        if self.current_step >= self.total_steps - 1:
            self.done = True
            current_usage_cost = self.price * max(self.grid, 0)
            reward = - (current_usage_cost / 1000)
        else:
            self.current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
            self.load = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'load'].iloc[0])
            self.pv = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'pv'].iloc[0])

            self.ess = action
            self.grid = self.load - self.pv + (self.ess / 4)

            self.price, self.peak_time = calculate_price(self.current_date)
            self.switch_sum += calculate_switch(action, self.cnd_state, self.current_step)
            self.peak = calculate_peak(self.peak, self.load, self.pv, action, self.contract)

            current_usage_cost = self.price * max(self.grid, 0)
            self.usage_sum += current_usage_cost
            vat = (self.peak * self.demand_cost + self.usage_sum) * 0.137
            self.total_cost = (self.peak * self.demand_cost + self.usage_sum) + vat

            self.soc = calculate_soc(self.soc, action, self.battery_cap)
            reward = - (current_usage_cost / 1000)

            self.current_step += 1


        next_state = [
            float(self.soc.item() if isinstance(self.soc, np.ndarray) else self.soc),
            float(self.load.item() if isinstance(self.load, np.ndarray) else self.load),
            float(self.pv.item() if isinstance(self.pv, np.ndarray) else self.pv),
            float(self.grid.item() if isinstance(self.grid, np.ndarray) else self.grid),
            float(self.total_cost.item() if isinstance(self.total_cost, np.ndarray) else self.total_cost),
            float(self.switch_sum.item() if isinstance(self.switch_sum, np.ndarray) else self.switch_sum),
            float(self.peak.item() if isinstance(self.peak, np.ndarray) else self.peak)
        ]

        return next_state, reward, self.done

    def render(self, action):
        print(f"Step: {self.current_step}")
        print(f"[{self.current_date}] SoC: {round(self.soc * 100, 2)}%")
        print(f"Load: {self.load}, PV: {self.pv}, Grid : {round(self.grid, 2)}, Price: {self.price}")
        print(f"Total: {round(self.total_cost, 0)}, Demand: {round(self.demand_cost * self.peak, 0)}, Usage: {round(self.usage_sum, 0)}")
        print(f"Switch Sum: {self.switch_sum}, Peak: {self.peak}")
        print(f'action: {action}')
        print("-" * 40)


def compute_reward(total_cost, peak, switch_sum, a=w1, b=w2, c=w3):
    total_cost = cost_normalize(total_cost, 0, 5000000)
    peak = peak_normalize(peak, 0, 500)
    switch_sum = switch_normalize(switch_sum, 0, 957)

    reward = - (a * total_cost + b * peak + c * switch_sum)
    return reward


def cost_normalize(data, min_val, max_val):
    if max_val == min_val:
        return 0
    return (data - min_val) / (max_val - min_val)


def switch_normalize(data, min_val, max_val):
    if max_val == min_val:
        return 0
    return (data - min_val) / (max_val - min_val)


def peak_normalize(data, min_val, max_val):
    if max_val == min_val:
        return 0
    return (data - min_val) / (max_val - min_val)


def calculate_soc(previous_soc, action, max_capacity):
    soc = ((max_capacity * previous_soc) + (action / 4)) / 1000
    return max(0, min(1, soc))


def calculate_peak(grid_prev, load, pv, action, P_contract):
    max_P_grid = max(grid_prev, load - pv + action)  # grid의 최대값 계산

    if max_P_grid > P_contract * 0.3:
        return max_P_grid
    else:
        return P_contract * 0.3


def calculate_switch(ess, cnd_state, current_step):
    if ess > 0:
        switch = 1
    elif ess < 0:
        switch = -1
    else:
        switch = 0

    previous_state = cnd_state[-1]
    if (previous_state == 1 and switch == -1) or (previous_state == -1 and switch == 1):
        switch_value = 2
    elif previous_state == switch:
        switch_value = 0
    else:
        switch_value = 1

    cnd_state.append(switch)
    return switch_value

def calculate_price(datetime_str):
    dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    dt_15m_ago = dt - datetime.timedelta(minutes=15)
    month = dt_15m_ago.month
    hour = dt_15m_ago.hour

    if 6 <= month <= 8:
        if 22 <= hour or hour <= 8:
            return 94, 0
        elif 8 <= hour <= 11 or 12 <= hour <= 13 or 18 <= hour <= 22:
            return 146.9, 1
        else:
            return 229, 2
    elif month in [11, 12, 1, 2]:
        if 22 <= hour or hour <= 8:
            return 101, 0
        elif 8 <= hour <= 9 or 12 <= hour <= 16 or 19 <= hour <= 22:
            return 147.1, 1
        else:
            return 204.6, 2
    else:
        if 22 <= hour or hour <= 8:
            return 94, 0
        elif 8 <= hour <= 11 or 12 <= hour <= 13 or 18 <= hour <= 22:
            return 116.5, 1
        else:
            return 147.2, 2

class PPOAgent:
    def __init__(self, state_size, action_size, env, gamma=0.995, learning_rate=0.0003, clip_ratio=0.2, lambd=0.97,
                 value_coef=0.5, entropy_coef=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.gamma = gamma  # 할인 인자, 미래 보상을 더 중시하도록 값 증가
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.lambd = lambd  # GAE의 lambda, 미래 보상을 더 고려하도록 값 증가
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lambd * (1 - dones[t]) * last_advantage
        returns = advantages + values
        return advantages, returns

    def _build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            tf.keras.layers.Dense(self.action_size, activation='tanh',
                                  kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003))
        ])
        return model

    def ppo_loss(self, old_policy, advantages, returns, actions, states):
        with tf.GradientTape() as tape:
            policy = self.actor(states)
            values = self.critic(states)

            mu = policy  # 액터 모델에서 나온 mu를 사용
            sigma = tf.math.softplus(tf.Variable(0.5))  # 표준편차 값, 필요에 따라 조정 가능

            # 액션 확률 분포의 로그 확률 계산
            log_probs_new = -0.5 * tf.square((actions - mu) / sigma)
            log_probs_old = -0.5 * tf.square((actions - old_policy) / sigma)

            # 중요도 샘플링 비율 계산
            ratio = tf.exp(tf.reduce_sum(log_probs_new - log_probs_old, axis=1))
            clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages

            # 손실 계산
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clip_adv))
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            entropy_loss = -tf.reduce_mean(tf.reduce_sum(-tf.square(policy), axis=1))
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss

        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        # 디버깅을 위해 손실과 그래디언트 확인
        tf.print("Actor Loss:", actor_loss)
        tf.print("Critic Loss:", critic_loss)
        tf.print("Gradients:", grads)

    def _build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        return model

    def select_action(self, state):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        tf.print("State Tensor:", state_tensor)
        mu = self.actor(state_tensor)

        print(f'MU : {mu}')

        soc, load, pv, grid, total_cost, switch_sum, peak = state[0]

        # 액션 노이즈 추가하여 탐험 강화
        dist = tf.random.normal(shape=mu.shape, mean=mu, stddev=2.0)  # stddev 값을 더 증가시켜 탐험 강화
        action2 = tf.clip_by_value(dist, -1, 1) * 250  # 클리핑 범위 유지
        print(f'초기 action : {action2}')
        # 액션 값 확인 및 SoC 제약 조건 적용
        action = action2.numpy()[0]

        # 다음 스텝의 예측 데이터를 사용
        next_step = self.env.current_step + 1
        next_load = float(self.env.data.loc[self.env.data['Unnamed: 0_x'] == next_step, 'load'].iloc[0])
        next_pv = float(self.env.data.loc[self.env.data['Unnamed: 0_x'] == next_step, 'pv'].iloc[0])

        # SoC 제약 조건 계산
        max_charge = min(self.env.conv_cap, (self.env.soc_max - soc) * self.env.battery_cap)
        max_discharge = min(self.env.conv_cap, (soc - self.env.soc_min) * self.env.battery_cap)

        # 충전/방전 후 예상 SoC 및 Grid 상태 계산
        expected_soc = soc + (action / 4) / self.env.battery_cap
        expected_grid = next_load - next_pv
        expected_grid_action = expected_grid + (action / 4)


        # 역송 방지 및 SoC 제약 조건 적용
        if expected_grid < 0:
            if soc <= self.env.soc_max:
                action = min(max_charge, (-4 * expected_grid) * 1.1)  # 역송 방지 위해 충전
            else:
                action = 0  # 충전 용량 없으면 대기
        elif expected_grid_action < 0:
            if expected_soc <= self.env.soc_min:
                max_safe_discharge = (soc - self.env.soc_min) * self.env.battery_cap * 4
                action = max(-max_safe_discharge, action, (-4 * expected_grid) * 0.9)
            else:
                action = max(-(4 * expected_grid) * 0.9, -max_discharge)
        else:
            if expected_soc < self.env.soc_min:
                required_action = (self.env.soc_min - soc) * self.env.battery_cap * 4
                action = max(required_action, action)
            elif expected_soc > self.env.soc_max:
                required_action = (self.env.soc_max - soc) * self.env.battery_cap * 4
                action = min(required_action, action)
            else:
                action = max(-max_discharge, min(action, max_charge))

        # 작은 액션 제거 (노이즈 방지)
        if -20 < action < 20:
            action = 0

        expected_soc = soc + (action / 4) / self.env.battery_cap
        expected_grid_action = expected_grid + (action / 4)
        print(
            f'Load/PV: {next_load}/{next_pv}, Next grid: {expected_grid}, Grid with act: {expected_grid_action}, Expected SoC: {expected_soc}')

        return action


    def train(self, states, actions, rewards, next_states, dones, old_policies):
        # 상태 데이터의 모양을 조정합니다.
        states = np.array(states, dtype=np.float32).reshape([-1, self.state_size])
        next_states = np.array(next_states, dtype=np.float32).reshape([-1, self.state_size])

        # actions을 1D 배열로 변환
        actions = np.array([action if np.isscalar(action) else action[0] for action in actions],
                           dtype=np.float32).reshape([-1, 1])

        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        old_policies = np.array(old_policies, dtype=np.float32).reshape([-1, self.action_size])

        values = self.critic(states)
        next_values = self.critic(next_states)
        advantages, returns = self.compute_advantages(rewards, values, next_values, dones)
        self.ppo_loss(old_policies, advantages, returns, actions, states)

def flatten_list(nested_list):
    """
    중첩된 리스트나 배열을 1차원 리스트로 평탄화하는 함수.
    """
    flat_list = []
    for sublist in nested_list:
        if isinstance(sublist, (list, np.ndarray)):
            flat_list.extend(flatten_list(sublist))
        else:
            flat_list.append(sublist)
    return flat_list

def plot_results(dates, load, pv, grid, action, soc):
    plt.figure(figsize=(14, 8))
    plt.plot(dates, load, label="Load", color="blue")
    plt.plot(dates, pv, label="PV", color="orange")
    plt.plot(dates, grid, label="Grid", color="green")

    plt.twinx()
    plt.plot(dates, soc, label="SoC", color="purple", linestyle='--')

    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.title("Load, PV, Grid, Action, and SoC Over Time")
    plt.legend(loc="upper left")
    plt.show()

def train_and_save_ppo(train_csv, num_episodes=1000, batch_size=64):
    episode_rewards = []
    env = Environment(data=train_csv)
    state_size = 7
    action_size = 1
    agent = PPOAgent(state_size, action_size, env)

    for e in range(num_episodes):
        states, actions, rewards, next_states, dones, old_policies = [], [], [], [], [], []
        state = env.reset()
        total_reward = 0

        for time in range(env.total_steps):
            state = np.reshape(state, [1, state_size])
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            old_policy = agent.actor(state_tensor).numpy()

            states.append(state)
            actions.append(action)
            # Ensure reward is a scalar
            if isinstance(reward, (list, np.ndarray)):
                reward = reward[0]
            rewards.append(float(reward))
            next_states.append(next_state)
            dones.append(done)
            old_policies.append(old_policy)

            state = next_state
            total_reward += reward

            if done:
                agent.train(
                    np.array(states, dtype=np.float32),
                    np.array(actions, dtype=np.float32),
                    np.array(rewards, dtype=np.float32),
                    np.array(next_states, dtype=np.float32),
                    np.array(dones, dtype=np.float32),
                    np.array(old_policies, dtype=np.float32)
                )
                # Ensure total_reward is a scalar
                if isinstance(total_reward, (list, np.ndarray)):
                    total_reward = total_reward[0]
                episode_rewards.append(total_reward)
                print(f"Episode: {e + 1}/{num_episodes}, Total Reward: {total_reward}")
                break

        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode vs Total Reward')
        plt.show()

        if e % 10 == 0:
            agent.actor.save(f'{e}_ppo_actor.keras')  # Updated to .keras format
            agent.critic.save(f'{e}_ppo_critic.keras')  # Updated to .keras format


def load_and_predict_ppo(model_path, csv_path, env_class, state_size, action_size):
    actor_model = tf.keras.models.load_model(f'../test/{model_path}_actor.h5')
    critic_model = tf.keras.models.load_model(f'../test/{model_path}_critic.h5')

    new_csv = csv_path
    new_env = env_class(data=new_csv)

    dates, load_values, pv_values, grid_values, actions, soc_values = [], [], [], [], [], []

    state = new_env.reset()
    total_reward = 0

    while True:
        state = np.reshape(state, [1, state_size])

        mu = actor_model(state)
        dist = tf.random.normal(shape=mu.shape, mean=mu, stddev=1.0)
        action = tf.clip_by_value(dist, -1, 1) * 250

        value = critic_model(state)
        print(f"Predicted value of current state: {value.numpy()[0]}")

        dates.append(new_env.current_date)
        load_values.append(new_env.load)
        pv_values.append(new_env.pv)
        grid_values.append(new_env.grid)
        actions.append(action.numpy()[0])
        soc_values.append(new_env.soc)

        next_state, reward, done = new_env.step(action.numpy()[0])
        total_reward += reward

        if done:
            print(f"Total Reward in new environment: {total_reward}")
            break
        state = next_state

    plot_results(dates, load_values, pv_values, grid_values, actions, soc_values)

    return total_reward

test_csv = pd.read_csv('../test/0907_0911.csv')
train_and_save_ppo(test_csv)
