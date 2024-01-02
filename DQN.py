import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import pandas as pd

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
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
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
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


data = pd.read_csv('pqms_data.csv')
# EMS 시뮬레이션 환경
class EMSEnvironment:
    def __init__(self, data):
        self.index = 0  # 인덱스
        self.state_of_charge = 50  # SOC 초기값
        self.load = 0  # 현재 부하
        self.pv = 0  # 현재 태양광 발전량
        self.price = 0 # 현재 요금
        # 다른 필요한 변수들을 추가할 수 있습니다.

    def reset(self):
        self.index = 0
        self.state_of_charge = 50
        self.load = 0
        self.pv = 0
        return np.array([self.state_of_charge, self.load, self.pv, self.index])

    def step(self, action):
        # 여기에 실제 시스템에 대한 행동의 영향을 계산하는 로직을 구현합니다.
        # 예: action이 -250 ~ -1이면 방전, 1 ~ 250이면 충전, 0이면 대기

        done = False
        reward = 0  # 보상 함수를 구현합니다.
        # 보상함수는 전기세로 grid * price가 가장 작은 값

        # 다음 상태 업데이트
        self.index += 1
        self.state_of_charge += action  # 예시: 행동에 따라 SOC 업데이트
        next_state = np.array([self.state_of_charge, self.load, self.pv, self.index])

        # 에피소드 종료 조건을 체크합니다 (예: 특정 시간에 도달하거나 SOC가 특정 범위를 벗어날 때)
        if self.index >= 24 or self.state_of_charge < 0 or self.state_of_charge > 100:
            done = True

        return next_state, reward, done

    def update_environment(self, current_time):
        self.index = current_time
        self.grid_usage = self.get_grid_usage(current_time)
        self.load = self.get_load(current_time)
        self.pv = self.get_pv_output(current_time)
        self.electricity_price = self.get_electricity_price(current_time)
        self.is_holiday = self.check_holiday(current_time)
        return

    def calculate_reward(self, action, state):
        # 보상 계산 로직 구현
        # 예: 전력 요금 절감, 효율적인 에너지 사용 등을 고려
        ...
        return reward


# DQN 에이전트와 EMS 환경 초기화
state_size = 4  # 상태 크기 (SOC, Load, PV, Time)
action_size = 501  # 행동 크기 (-250 ~ 250)
agent = DQNAgent(state_size, action_size)
env = EMSEnvironment()

# 학습 루프
for e in range(50):  # 에피소드 수
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(24):  # 한 에피소드의 최대 길이
        action = agent.act(state) - 250  # 행동 선택 및 조정
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"에피소드: {e}/{1000}, 시간: {time}, 보상: {reward}")
            break
        if len(agent.memory) > 32:
            agent.replay(32)
