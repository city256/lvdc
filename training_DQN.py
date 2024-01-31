import datetime
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from datetime import timedelta
import db_fn
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

total_episodes = 100
batch_size = 32
start_time = (datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))).strftime('%Y-%m-%d %H:%M:%S')


def calculate_price(datetime_str):
    dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    month = dt.month
    hour = dt.hour

    if 6 <= month <= 8: # 여름철
        if 22 <= hour or hour <= 8: # 경부하
            return 94
        elif 8 <= hour <= 11 or 12 <= hour <= 13 or 18 <= hour <= 22: # 중간부하
            return 146.9
        else: # 최대부하
            return 229
    elif month in [11, 12, 1, 2]: # 겨울철
        if 22 <= hour or hour <= 8:  # 경부하
            return 101
        elif 8 <= hour <= 9 or 12 <= hour <= 16 or 19 <= hour <= 22:  # 중간부하
            return 147.1
        else:  # 최대부하
            return 204.6
    else :  # 봄, 가을철
        if 22 <= hour or hour <= 8:  # 경부하
            return 94
        elif 8 <= hour <= 11 or 12 <= hour <= 13 or 18 <= hour <= 22:  # 중간부하
            return 116.5
        else:  # 최대부하
            return 147.2

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        current_grid = state[0][1] - state[0][2]
        current_soc = state[0][0]

        if current_soc <= 10:
            valid_actions = [i for i in range(251)]  # 충전만 가능 (0 ~ 250)
        elif current_soc >= 90:
            valid_actions = [i for i in range(-250, 1) if i <= current_grid * 4]  # 방전만 가능 (-250 ~ 0)
        else:
            valid_actions = [i for i in range(-250, 251) if i <= current_grid * 4]  # 모든 행동 가능

        if not valid_actions:  # 유효한 행동이 없는 경우
            return 0  # 대기 행동을 반환

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)  # 탐험
        else:
            act_values = self.model.predict(state)
            # 유효한 행동들 중에서 최대 Q 값을 가지는 행동을 선택
            return max(valid_actions, key=lambda x: act_values[0][x + 250])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        #self.model.save_weights(name)
        self.model.save(name)

# EMS 시뮬레이션 환경
class EMSEnvironment:
    def __init__(self):
        self.data = pd.read_csv('test/merged_data.csv')
        self.current_index = 0
        self.current_time = (datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))).strftime('%Y-%m-%d %H:%M:00')
        self.max_index = 0
        self.price = calculate_price(self.current_time)
        self.soc = 50
        self.load = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_index, 'load'].iloc[0], 2))
        self.pv = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_index, 'pv'].iloc[0], 2))
        # 다른 필요한 변수들을 추가할 수 있습니다.

    def reset(self):
        self.current_index = self.data.loc[self.data['date'] == (datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))).strftime(
            '%Y-%m-%d %H:%M:00'), 'Unnamed: 0_x'].iloc[0]
        self.max_index = len(self.data)

        self.current_time = (datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))).strftime('%Y-%m-%d %H:%M:00')
        self.price = calculate_price(self.current_time)
        self.sum_discharge=0
        self.sum_charge = 0
        self.sum_grid = 0
        self.sum_fee = 0
        self.soc = db_fn.get_pms_soc()
        self.load = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_index, 'load'].iloc[0], 2))
        self.pv = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_index, 'pv'].iloc[0], 2))
        return np.array([self.soc, self.load, self.pv, self.price])

    def step(self, action):

        self.load = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_index, 'load'].iloc[0], 2))
        self.pv = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_index, 'pv'].iloc[0], 2))
        # 여기에 실제 시스템에 대한 행동의 영향을 계산하는 로직을 구현합니다.
        done = False
        charge = 0
        discharge = 0
        # 예: action이 -250 ~ -1이면 방전, 1 ~ 250이면 충전, 0이면 대기
        if action > 0:
            charge = action/4
        else:
            discharge = action/4

        # 보상함수 계산 (전기세 계산)
        grid = self.load - self.pv - discharge + charge
        reward = -grid * self.price


        # 다음 상태 업데이트
        self.current_index += 1
        self.soc += (action/4) * 0.1  # 예시: 행동에 따라 SOC 업데이트
        self.load = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_index, 'load'].iloc[0], 2))
        self.pv = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_index, 'pv'].iloc[0], 2))
        self.sum_discharge += discharge
        self.sum_charge += charge
        self.sum_grid += grid
        self.sum_fee += reward
        next_state = np.array([self.soc, self.load, self.pv, self.price])

        # 에피소드 종료 조건을 체크합니다 (예: 24시간을 넘어갔을때)
        if self.current_index >= self.max_index - 1:
            done = True
        return next_state, reward, done


# DQN 에이전트와 EMS 환경 초기화
state_size = 4  # 상태 크기 (SOC, Load, PV, Price)
action_size = 501  # 행동 크기 (-250 ~ 250)
agent = DQNAgent(state_size, action_size)
env = EMSEnvironment()

'''
# 학습 루프
for e in range(20):  # 에피소드 수
    print('episode = ', e)
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(96):  # 한 에피소드의 최대 길이
        #print('time index = ', time, 'state= ', state, 'index=', env.current_index, env.load, env.pv)
        action = agent.act(state)  # 행동 선택 및 조정
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"에피소드: {e}/{1000}, 시간: {time}, 보상: {env.sum_fee}, 충전: {env.sum_charge}, 방전: {env.sum_discharge}, soc: {env.soc}, grid: {env.sum_grid}")
            break
        if len(agent.memory) > 32:
            agent.replay(32)
'''
model_save_path = 'dqn_model.h5'

#agent.save(model_save_path)
model = load_model(model_save_path)

simulation_file = 'csv/pqms_data_15_dqn.csv'
sim_data = pd.read_csv(simulation_file)


