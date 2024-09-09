import pandas as pd
import datetime
import tensorflow as tf
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

train_csv = pd.read_csv('../test/train_1day.csv')
test_csv = pd.read_csv('../test/testset.csv')

# hyperparameter
w1 = 1 # cost weight
w2 = 0 # peak weight
w3 = 0 # switch weight

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

        # 모델링 변수 설정
        self.current_step = 0
        self.current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        self.done = False
        self.soc = 0.5  # 초기 SoC 설정 (충전 상태)
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

        self.usage_cost = max(0, self.grid) * self.price
        self.usage_sum = 0
        self.total_cost = (self.peak * self.demand_cost + self.usage_sum) * 0.137
        self.total_costs = []

    def reset(self):
        self.current_step = 0
        self.done = False
        self.current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        self.soc = 0.5
        self.load = float(np.round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'load'].iloc[0], 2))
        self.pv = float(np.round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'pv'].iloc[0], 2))
        self.price, self.peak_time = calculate_price(self.current_date)
        self.charge = 0
        self.discharge = 0
        self.ess = self.charge - self.discharge
        self.grid = self.load - self.pv + self.ess
        self.peak = 0
        self.switch = 0
        self.switch_sum = 0
        self.usage_sum = 0
        vat = (self.peak * self.demand_cost + self.usage_sum) * 0.137
        self.total_cost = (self.peak * self.demand_cost + self.usage_sum) + vat
        self.cnd_state = [0]
        self.total_costs = []

        # 추가된 상태 정보
        state = [self.soc, self.load, self.pv, self.grid, self.total_cost, self.switch_sum, self.peak]
        return state

    def step(self, action):
        """
        주어진 액션에 따라 환경의 상태를 업데이트하고,
        다음 상태, 보상, 에피소드 종료 여부를 반환합니다.
        """
        if self.current_step >= self.total_steps - 2:  # 마지막 스텝에서 종료
            self.done = True
            reward = np.array([float(0)])  # 마지막 스텝에서는 더 이상 보상이 없으므로 0으로 설정
        else:
            # 현재 스텝 상태 업데이트
            self.current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
            self.load = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'load'].iloc[0])
            self.pv = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'pv'].iloc[0])

            # ESS 상태 업데이트
            self.ess = action
            self.grid = self.load - self.pv + (self.ess / 4)

            # 현재 스텝의 가격과 피크 시간 계산
            self.price, self.peak_time = calculate_price(self.current_date)

            # 충방전 스위치 상태 업데이트
            self.switch_sum += calculate_switch(action, self.cnd_state, self.current_step)

            # 현재 스텝에서의 피크 계산
            self.peak = calculate_peak(self.peak, self.load, self.pv, action, self.contract)

            # 현재 스텝에서의 사용 비용 계산
            self.usage_sum += self.price * self.grid
            vat = (self.peak * self.demand_cost + self.usage_sum) * 0.137
            self.total_cost = (self.peak * self.demand_cost + self.usage_sum) + vat

            # SoC 업데이트 (충전 또는 방전)
            self.soc = calculate_soc(self.soc, action, self.battery_cap)

            # 보상 계산 (정규화 적용)
            reward = -self.usage_sum/100000

            # 현재 스텝을 증가
            self.current_step += 1

        # 다음 상태 계산
        next_state = np.array(
            [float(self.soc), float(self.load), float(self.pv), float(self.grid), float(self.total_cost),
             float(self.switch_sum), float(self.peak)],
            dtype=np.float32)

        if self.done:
            print(f'soc: {self.soc}, peak: {self.peak}, switch_sum: {self.switch_sum}')

        self.render(action)

        return next_state, reward, self.done

    def render(self, action):
        # 현재 상태를 출력
        print(f"Step: {self.current_step}")
        print(f"[{self.current_date}] SoC: {np.round(self.soc, 2) * 100}%")
        print(f"Load: {self.load}, PV: {self.pv}, Grid: {np.round(self.grid, 2)}")
        print(
            f"Total: {np.round(self.total_cost, 0)}, Demand: {np.round(self.demand_cost * self.peak, 0)}, Usage: {np.round(self.usage_sum, 0)}")
        print(f"Switch Sum: {self.switch_sum}, Peak: {self.peak}")
        print(f'action: {action}')
        print("-" * 40)

    # 필요한 함수 정의 (앞서 정의된 함수들)
def calculate_soc(previous_soc, action, max_capacity):
    soc = ((max_capacity * previous_soc) + (action / 4)) / 1000
    return max(0, min(1, soc))

def calculate_peak(grid_prev, load, pv, action, P_contract):
    max_P_grid = max(grid_prev, load - pv + action)  # grid의 최대값 계산
    return max_P_grid if max_P_grid > P_contract * 0.3 else P_contract * 0.3

def calculate_switch(ess, cnd_state, current_step):
    switch = 1 if ess > 0 else -1 if ess < 0 else 0
    previous_state = cnd_state[-1]

    if (previous_state == 1 and switch == -1) or (previous_state == -1 and switch == 1):
        switch_value = 2  # 충전에서 방전으로, 방전에서 충전으로 전환
    elif previous_state == switch:
        switch_value = 0  # 상태 변화 없음
    else:
        switch_value = 1  # 대기 상태로의 전환 또는 대기에서 충전/방전으로 전환

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

class TD3Agent:
    def __init__(self, state_size, action_size, action_bound, env):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.env = env  # 환경 인스턴스를 저장
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.tau = 0.005
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.exploration_prob = 0.3  # 탐험할 확률 추가

        self.actor, self.critic1, self.critic2 = self.build_networks()
        self.actor_target, self.critic1_target, self.critic2_target = self.build_networks()
        self.update_target_networks(tau=1.0)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    def build_networks(self):
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        dense1 = tf.keras.layers.Dense(400, activation='relu')(state_input)
        dense2 = tf.keras.layers.Dense(300, activation='relu')(dense1)
        out_mu = tf.keras.layers.Dense(self.action_size, activation='tanh')(dense2)
        actor = tf.keras.Model(state_input, out_mu)

        action_input = tf.keras.layers.Input(shape=(self.action_size,))
        state_action_input = tf.keras.layers.Concatenate()([state_input, action_input])
        dense1 = tf.keras.layers.Dense(400, activation='relu')(state_action_input)
        dense2 = tf.keras.layers.Dense(300, activation='relu')(dense1)
        q_value = tf.keras.layers.Dense(1, activation='linear')(dense2)
        critic1 = tf.keras.Model([state_input, action_input], q_value)
        critic2 = tf.keras.Model([state_input, action_input], q_value)

        return actor, critic1, critic2

    def update_target_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        new_weights = []
        for target_weights, weights in zip(self.actor_target.weights, self.actor.weights):
            new_weights.append(tau * weights + (1 - tau) * target_weights)
        self.actor_target.set_weights(new_weights)

        for target, source in zip([self.critic1_target, self.critic2_target], [self.critic1, self.critic2]):
            new_weights = []
            for target_weights, weights in zip(target.weights, source.weights):
                new_weights.append(tau * weights + (1 - tau) * target_weights)
            target.set_weights(new_weights)

    def select_action(self, state):
        if np.random.rand() < self.exploration_prob:
            # Random action for exploration
            action = np.random.uniform(-self.action_bound, self.action_bound, self.action_size)
        else:
            # Deterministic action based on the policy
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            mu = self.actor(state_tensor)
            mu = mu.numpy()[0]

            # Apply noise
            noise = np.random.normal(0, self.policy_noise, size=self.action_size)
            action = mu + noise
            action = np.clip(action, -self.action_bound, self.action_bound)

        action = self.apply_constraints(state, action)
        return action

    def apply_constraints(self, current_state, action):
        soc, load, pv, grid, total_cost, switch_sum, peak = current_state[0]

        next_step = self.env.current_step
        next_load = float(self.env.data.loc[self.env.data['Unnamed: 0_x'] == next_step, 'load'].iloc[0])
        next_pv = float(self.env.data.loc[self.env.data['Unnamed: 0_x'] == next_step, 'pv'].iloc[0])

        max_charge = min(self.env.conv_cap, (self.env.soc_max - soc) * self.env.battery_cap)
        max_discharge = min(self.env.conv_cap, (soc - self.env.soc_min) * self.env.battery_cap)

        expected_soc = soc + (action / 4) / self.env.battery_cap
        expected_grid = next_load - next_pv
        expected_grid_action = expected_grid + (action / 4)

        if expected_grid < 0:
            if soc <= self.env.soc_max:
                action = min(max_charge, (-4 * expected_grid) * 1.1)
            else:
                action = 0
        else:
            if expected_grid_action < 0:
                if expected_soc <= self.env.soc_min:
                    max_safe_discharge = (soc - self.env.soc_min) * self.env.battery_cap * 4
                    action = max(-max_safe_discharge * 0.9, -max_discharge, (-4 * expected_grid) * 0.9)
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

        # Check if the action is too small due to constraints
        if -20 < action < 20:
            action = np.random.uniform(-max_discharge, max_charge)  # Add random exploration if action is too small

        return action

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        minibatch = random.sample(replay_buffer, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = state.reshape(1, -1)
            next_state = next_state.reshape(1, -1)

            # Ensure action is a numpy array with the correct shape
            if isinstance(action, float):
                action = np.array([action])

            action = tf.convert_to_tensor(action.reshape(1, -1), dtype=tf.float32)
            next_action = self.actor_target(next_state)

            noise = np.clip(np.random.normal(0, self.policy_noise, size=self.action_size), -self.noise_clip,
                            self.noise_clip)
            next_action = next_action.numpy() + noise
            next_action = np.clip(next_action, -self.action_bound, self.action_bound)

            target_q1 = self.critic1_target([next_state, next_action])
            target_q2 = self.critic2_target([next_state, next_action])
            target_q = reward + (1 - done) * self.gamma * np.minimum(target_q1, target_q2)

            with tf.GradientTape() as tape:
                q1 = self.critic1([state, action])
                loss1 = tf.keras.losses.MSE(target_q, q1)
            critic_grads1 = tape.gradient(loss1, self.critic1.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads1, self.critic1.trainable_variables))

            with tf.GradientTape() as tape:
                q2 = self.critic2([state, action])
                loss2 = tf.keras.losses.MSE(target_q, q2)
            critic_grads2 = tape.gradient(loss2, self.critic2.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads2, self.critic2.trainable_variables))

            if self.policy_delay % 2 == 0:
                with tf.GradientTape() as tape:
                    new_actions = self.actor(state)
                    critic_val = self.critic1([state, new_actions])
                    actor_loss = -tf.reduce_mean(critic_val)
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

                self.update_target_networks()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        self.train(self.memory, batch_size)

    def load(self, name):
        self.actor = tf.keras.models.load_model(f"{name}_actor.h5")
        self.critic1 = tf.keras.models.load_model(f"{name}_critic1.h5")
        self.critic2 = tf.keras.models.load_model(f"{name}_critic2.h5")
        self.actor_target = tf.keras.models.load_model(f"{name}_actor_target.h5")
        self.critic1_target = tf.keras.models.load_model(f"{name}_critic1_target.h5")
        self.critic2_target = tf.keras.models.load_model(f"{name}_critic2_target.h5")

    def save(self, name):
        self.actor.save(f"{name}_actor.h5")
        self.critic1.save(f"{name}_critic1.h5")
        self.critic2.save(f"{name}_critic2.h5")
        self.actor_target.save(f"{name}_actor_target.h5")
        self.critic1_target.save(f"{name}_critic1_target.h5")
        self.critic2_target.save(f"{name}_critic2_target.h5")

def plot_results(dates, load, pv, grid, action, soc):
    plt.figure(figsize=(14, 8))

    # Load, PV, Grid 값을 하나의 그래프에 플롯
    plt.plot(dates, load, label="Load", color="blue")
    plt.plot(dates, pv, label="PV", color="orange")
    plt.plot(dates, grid, label="Grid", color="green")

    # Action과 SoC는 오른쪽 y축을 사용하여 플롯
    plt.twinx()
    plt.plot(dates, soc, label="SoC", color="purple", linestyle='--')

    # 라벨 및 타이틀 설정
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.title("Load, PV, Grid, and SoC Over Time")

    # 범례 추가
    plt.legend(loc="upper left")
    plt.show()

def train_and_save_model(train_csv):
    episode_rewards = []
    env = Environment(data=train_csv)

    state_size = 7  # State space size
    action_size = 1  # Action space size
    action_bound = 250
    agent = TD3Agent(state_size, action_size, action_bound, env)

    num_episodes = 1000  # Number of episodes
    batch_size = 64

    for e in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(env.total_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            # Flatten next_state if it contains nested structures
            if any(isinstance(i, (list, np.ndarray)) for i in next_state):
                next_state = np.hstack(next_state).astype(np.float32)
            else:
                next_state = np.array(next_state, dtype=np.float32)

            if next_state.ndim == 1:
                next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                agent.update_target_networks()  # Update target networks
                print(f"Episode: {e}/{num_episodes}, Total Reward: {total_reward}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)  # Replay and train

        agent.save(f'td3_train{e}_v1.keras')

        episode_rewards.append(total_reward)

        # Plot the reward progress
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode vs Total Reward')
        plt.show()

def load_and_predict(test_csv, model_load_path, epsilon=0):
    new_env = Environment(data=test_csv)

    state_size = 7
    action_size = 1
    action_bound = 250
    agent = TD3Agent(state_size, action_size, action_bound, new_env)
    agent.load(model_load_path)

    dates, loads, pvs, grids, actions, socs = [], [], [], [], [], []

    state = new_env.reset()
    state = np.reshape(state, [1, len(state)])

    done = False
    total_reward = 0

    while not done:
        dates.append(new_env.current_date)
        loads.append(new_env.load)
        pvs.append(new_env.pv)
        grids.append(new_env.grid)
        socs.append(new_env.soc)

        action = agent.select_action(state)
        actions.append(action)
        next_state, reward, done = new_env.step(action)
        next_state = np.reshape(next_state, [1, len(next_state)])
        total_reward += reward
        state = next_state

    print(f"Total Reward for the new data: {total_reward}")

    plot_results(dates, loads, pvs, grids, actions, socs)

def load_and_resume_training(model_load_path, start_episode, num_episodes, train_csv):
    env = Environment(data=train_csv)

    state_size = 7
    action_size = 1
    action_bound = 250
    agent = TD3Agent(state_size, action_size, action_bound, env)
    agent.load(model_load_path)

    episode_rewards = []

    batch_size = 64

    for e in range(start_episode, start_episode + num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(env.total_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        if len(agent.memory) > batch_size:
            agent.train(agent.memory, batch_size)

        agent.save(f'td3_model_{e}')

        episode_rewards.append(total_reward)

        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode vs Total Reward')
        plt.show()

# Example usage
train_0711 = pd.read_csv('../test/0907_0911.csv')

train_and_save_model(train_0711)
# load_and_predict(test_csv, 'td3_model_50')
# load_and_resume_training('td3_model_49', 49, 100, train_csv)
