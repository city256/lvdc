import datetime
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from datetime import timedelta
import db_fn
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
        if np.random.rand() <= self.epsilon:
            # 탐험: 무작위 행동 선택
            return random.randrange(self.action_size) - 250
        else:
            # 활용: 신경망을 사용하여 행동 선택
            act_values = self.model.predict(state)
            return np.argmax(act_values[0]) - 250

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
        self.model.save_weights(name)

# EMS 시뮬레이션 환경
class EMSEnvironment:
    value = 2341
    def __init__(self):
        pv_date = (datetime.datetime.now()).strftime('%Y-%m-%d %H:00:00')
        load_date = (datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))).strftime(
            '%Y-%m-%d %H:%M:00')
        pv = pd.read_csv('pred_pv.csv')
        load = pd.read_csv('pred_load.csv')
        self.time = load_date
        self.price = calculate_price(load_date)
        self.current_time_load = pd.to_datetime(load_date).timestamp()
        self.current_time_pv = pv_date
        self.soc = db_fn.get_pms_soc()
        self.load = float(round(load.loc[load['date'] == load_date, 'load'].iloc[0], 2))
        self.pv = float(round(pv.loc[pv['date'] == pv_date, 'pv'].iloc[0] / 4, 2))
        # 다른 필요한 변수들을 추가할 수 있습니다.

    def reset(self):
        self.pv_date = (datetime.datetime.now()).strftime('%Y-%m-%d %H:00:00')
        self.load_date = (datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))).strftime(
            '%Y-%m-%d %H:%M:00')
        pv = pd.read_csv('pred_pv.csv')
        load = pd.read_csv('pred_load.csv')
        self.time = self.load_date
        self.price = calculate_price(self.load_date)
        self.current_time_load = pd.to_datetime(self.load_date).timestamp()
        self.current_time_pv = self.pv_date
        self.soc = db_fn.get_pms_soc()
        self.load = float(round(load.loc[load['date'] == self.load_date, 'load'].iloc[0], 2))
        self.pv = float(round(pv.loc[pv['date'] == self.pv_date, 'pv'].iloc[0] / 4, 2))
        return np.array([self.soc, self.load, self.pv, self.price])

    def step(self, action):
        # 여기에 실제 시스템에 대한 행동의 영향을 계산하는 로직을 구현합니다.
        print(self.current_time_load, self.soc, self.load, self.pv)
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
        self.current_time_load = (datetime.datetime.strptime(self.time, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
        self.soc += (action/4) * 0.1  # 예시: 행동에 따라 SOC 업데이트
        self.load_date = (datetime.datetime.strptime(self.time, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')

        next_state = np.array([self.soc, self.load, self.pv, self.price])

        # 에피소드 종료 조건을 체크합니다 (예: 24시간을 넘어갔을때)
        time_diff = datetime.datetime.strptime(self.current_time_load, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        if time_diff >= timedelta(days=1):
            print("wrong")
            done = True

        return next_state, reward, done

# DQN 에이전트와 EMS 환경 초기화
state_size = 4  # 상태 크기 (SOC, Load, PV, Price)
action_size = 501  # 행동 크기 (-250 ~ 250)
agent = DQNAgent(state_size, action_size)
env = EMSEnvironment()


# 학습 루프
for e in range(20):  # 에피소드 수
    print('episode = ', e)
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(96):  # 한 에피소드의 최대 길이
        print('time index = ', time, 'state= ', state)
        action = agent.act(state)  # 행동 선택 및 조정
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"에피소드: {e}/{1000}, 시간: {time}, 보상: {reward}")
            break
        if len(agent.memory) > 32:
            agent.replay(32)
