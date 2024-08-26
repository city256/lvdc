import pandas as pd
import datetime
from tqdm.keras import TqdmCallback
import tensorflow as tf
import numpy as np
import random
from collections import deque


# print('numpy version:', np.__version__)
# print('pnada version:', pd.__version__)
# print("TensorFlow version:", tf.__vers_ion__)
# print("GPU Available:", tf.test.is_gpu_available())
# print("GPU Device Name:", tf.test.gpu_device_name())

csv_file = pd.read_csv('../test/merged_data.csv')

# hyperparameter
w1 = 0.33 # cost weight
w2 = 0.33 # peak weight
w3 = 0.34 # swtich weight



class Environment:
    def __init__(self, data):
        # 초기 설정 상수
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

        self.usage_cost = self.grid * self.price
        self.usage_sum = 0
        self.total_cost = (self.peak * self.demand_cost + self.usage_sum) * 0.137


    def reset(self):
        """
        환경을 초기 상태로 재설정하고, 초기 상태를 반환합니다.
        """
        self.current_step = 0
        self.current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        self.done = False
        self.soc = 0.5  # 초기 SoC 설정
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

        state = [self.soc, self.load, self.pv, self.grid,  self.total_cost, self.switch_sum, self.peak]
        return state

    def step(self, action):
        """
        주어진 액션에 따라 환경의 상태를 업데이트하고,
        다음 상태, 보상, 에피소드 종료 여부를 반환합니다.
        """

        if self.current_step >= self.total_steps - 1:
            self.done = True
        else:
            self.current_step += 1

        if action > 0:
            charge = action
            discharge = 0
        elif action < 0:
            discharge = action
            charge = 0
        else:
            charge = 0
            discharge = 0

        ess = action

        # 현재 스텝의 데이터 가져오기
        current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        load = self.data.iloc[self.current_step]['load']
        pv = self.data.iloc[self.current_step]['pv']
        grid = load - pv + ess
        price, peak_time = calculate_price(current_date)
        self.switch_sum += calculate_switch(action, self.cnd_state, self.current_step)

        self.peak = calculate_peak(self.peak, grid, self.contract)
        usage_cost = price * grid
        self.usage_sum =self.usage_sum + usage_cost
        total_cost = (self.peak * self.demand_cost + self.usage_sum) * 0.137

        # SoC 업데이트 (충전 또는 방전)
        self.soc = calculate_soc(self.soc, action, self.battery_cap)
        next_state = [self.soc, load, pv, grid, total_cost, self.switch_sum, self.peak]

        # 보상 계산
        reward = compute_reward(total_cost, self.switch_sum, self.peak)


        if self.done:
            print(f'soc: {self.soc}, peak: {self.peak}, switch_sum: {self.switch_sum}')
            print(len(self.cnd_state), self.total_steps)
        return next_state, reward, self.done

    def render(self):
        # 현재 상태를 출력
        print(f"Step: {self.current_step}")
        print(f"SoC: {self.soc}")
        print(f"Load: {self.load}")
        print(f"PV: {self.pv}")
        print(f"Grid: {self.grid}")
        print(f"Total Cost: {self.total_cost}")
        print(f"Switch Sum: {self.switch_sum}")
        print(f"Peak: {self.peak}")
        print("-" * 40)


# 필요한 함수 정의 (앞서 정의된 함수들)
def calculate_soc(previous_soc, action, max_capacity):
    soc = ((max_capacity * previous_soc) + (action/4))/1000
    return max(0, min(1, soc))

def compute_reward(total_cost, peak, switch_sum, a=w1, b=w2, c=w3):
    reward = - (a * total_cost + b * peak + c * switch_sum)
    return reward

def calculate_peak(grid_prev, grid, P_contract):
    max_P_grid = max(grid_prev, grid)  # grid의 최대값 계산

    if max_P_grid > P_contract * 0.3:
        return max_P_grid
    else:
        return P_contract * 0.3

def calculate_switch(ess, cnd_state, current_step):
    if ess > 0:
        switch =  1
    elif ess < 0:
        switch = -1
    else:
        switch = 0

    # 이전 상태와 현재 상태 비교
    previous_state = cnd_state[-1]

    if (previous_state == 1 and switch == -1) or (previous_state == -1 and switch == 1):
        switch_value = 2  # 충전에서 방전으로, 방전에서 충전으로 전환
    elif previous_state == switch:
        switch_value = 0  # 상태 변화 없음
    else:
        switch_value = 1  # 대기 상태로의 전환 또는 대기에서 충전/방전으로 전환

    cnd_state.append(switch)

    #print(cnd_state[current_step], cnd_state[current_step-1],cnd_state[current_step] - cnd_state[current_step - 1] )
    return switch_value

def calculate_price(datetime_str):
    dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    dt_15m_ago = dt - datetime.timedelta(minutes=15)
    month = dt_15m_ago.month
    hour = dt_15m_ago.hour

    if 6 <= month <= 8: # 여름철
        if 22 <= hour or hour <= 8: # 경부하
            return 94, 0
        elif 8 <= hour <= 11 or 12 <= hour <= 13 or 18 <= hour <= 22: # 중간부하
            return 146.9, 1
        else: # 최대부하
            return 229, 2
    elif month in [11, 12, 1, 2]: # 겨울철
        if 22 <= hour or hour <= 8:  # 경부하
            return 101, 0
        elif 8 <= hour <= 9 or 12 <= hour <= 16 or 19 <= hour <= 22:  # 중간부하
            return 147.1, 1
        else:  # 최대부하
            return 204.6, 2
    else :  # 봄, 가을철
        if 22 <= hour or hour <= 8:  # 경부하
            return 94, 0
        elif 8 <= hour <= 11 or 12 <= hour <= 13 or 18 <= hour <= 22:  # 중간부하
            return 116.5, 1
        else:  # 최대부하
            return 147.2, 2

def test_environment_with_random_actions(env, num_episodes=1, max_steps=96):
    for episode in range(num_episodes):
        print(f"Episode {episode + 1} / {num_episodes}")
        state = env.reset()
        done = False
        step_count = 0

        while not done :
            # 랜덤한 액션 선택 (충전량은 -250 ~ 250 사이의 값으로 가정)
            random_action = np.random.uniform(-250, 250)

            # 스텝 실행
            next_state, reward, done = env.step(random_action)

            # 상태 및 보상 출력
            print(
                f"Step {step_count + 1}: Action = {random_action:.2f}, Next State = {next_state}, Reward = {reward:.2f}, Done = {done}")

            # 상태 업데이트
            state = next_state
            step_count += 1

        print(f"End of Episode {episode + 1}\n")

class DQNAgent:
    def __init__(self, state_size, action_size, env):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.env = env  # Environment 클래스 인스턴스 참조
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # 타겟 네트워크 초기화

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.state_size,)))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='tanh'))  # [-1, 1] 범위로 출력
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # 타겟 네트워크의 가중치를 현재 네트워크의 가중치로 업데이트
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.uniform(-250, 250)  # 가능한 액션 범위 내에서 랜덤 선택
        else:
            act_value = self.model.predict(state)[0][0]
            action = act_value * 250  # [-1, 1] 범위를 [-250, 250]으로 조정

        # 제약 조건 적용
        action = self.apply_constraints(state, action)
        return action

    def apply_constraints(self, state, action):
        soc, load, pv, grid, total_cost, switch_sum, peak = state[0]  # 상태를 분리

        # SoC 제약 조건 적용
        max_charge = min(self.env.conv_cap, (self.env.soc_max - soc) * self.env.battery_cap)
        max_discharge = min(self.env.conv_cap, (soc - self.env.soc_min) * self.env.battery_cap)

        # 충전과 방전 간 상호 배타적 조건 적용
        if action > 0:
            action = min(action, max_charge)  # 충전량은 최대 충전 가능량으로 제한
        elif action < 0:
            action = max(action, -max_discharge)  # 방전량은 최대 방전 가능량으로 제한

        # 그리드 전력 이동 금지 조건 적용
        expected_grid = load - pv + action
        if expected_grid < 0:
            action = -(load - pv)  # 그리드가 음수로 가지 않도록 조정

        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][0] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

env = Environment(data=csv_file)

# 예시로 DQNAgent를 초기화하고 사용할 수 있는 코드
state_size = 7  # 상태 크기 맞춤
action_size = 250  # 액션 공간 크기 (충전/방전 범위에 따라 다름)
agent = DQNAgent(state_size, action_size, env)

# Reset 환경에서 반환되는 state의 크기를 확인
state = env.reset()
print(f"State shape: {np.array(state).shape}")  # 크기 확인

# 에이전트와 환경 설정
state_size = 7  # 상태 공간 크기
action_size = 1  # 연속적인 액션 공간이지만 네트워크는 하나의 출력을 가짐
agent = DQNAgent(state_size, action_size, env)

# 학습 루프
num_episodes = 10  # 에피소드 수
batch_size = 32

for e in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0  # 에피소드별로 누적 보상을 초기화

    for time in range(env.total_steps):
        action = agent.act(state)  # 행동 선택
        next_state, reward, done = env.step(action)  # 환경에서 행동 수행
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)  # 경험 저장
        state = next_state  # 상태 업데이트

        total_reward += reward  # 보상 누적
        print(f'episode: {e}, step: {time}')
        env.render()

        if done:
            agent.update_target_model()  # 타겟 네트워크 업데이트
            print(f"Episode: {e + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)  # 리플레이에서 학습

    # 매 에피소드 종료 후 모델 저장 (원하는 경우)
    agent.model.save(f'dqn_model_{e}.keras')

