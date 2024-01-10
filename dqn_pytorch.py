import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import time
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randint(-250, 250)
        state = torch.from_numpy(state).float().unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item() - 250  # returns action

    def learn(self, state, action, reward, next_state, done):
        target = reward + self.gamma * torch.max(self.model(torch.from_numpy(next_state).float())).item() * (not done)
        target_f = self.model(torch.from_numpy(state).float())
        target_f[0][action + 250] = target
        loss = torch.nn.functional.mse_loss(target_f, target_f)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



class Environment:
    def __init__(self, data):
        self.data = data
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        # 보상 계산 로직 구현
        reward = ...  # 보상 계산
        self.current_step += 1
        done = self.current_step == len(self.data) - 1
        next_state = self.data.iloc[self.current_step].values if not done else np.zeros(self.data.shape[1])
        return next_state, reward, done

# CSV 파일 로드
data = pd.read_csv('your_data.csv')

# 환경 및 에이전트 초기화
state_size = data.shape[1]
action_size = 501  # -250부터 250까지
agent = DQNAgent(state_size, action_size)
env = Environment(data)
total_episodes = 20

# 에피소드 수행
for e in range(total_episodes):
    state = env.reset()

    for time in range(500):  # 또는 데이터의 길이에 따라 조정
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
# 모델 저장
torch.save(agent.model.state_dict(), 'dqn_model.pth')

# 학습 모델 불러와서 새로운 데이터에 적용
# 모델 로드
model = DQNNetwork(state_size, action_size)
model.load_state_dict(torch.load('dqn_model.pth'))
model.eval()

# 데이터 로드
data = pd.read_csv('pqms_data.csv')

# 액션 예측 및 결과 저장
actions = []
for index, row in data.iterrows():
    state = row.values
    state = torch.from_numpy(state).float().unsqueeze(0)
    with torch.no_grad():
        action_values = model(state)
        action = torch.argmax(action_values).item() - 250
    actions.append(action)

# 예측된 액션을 새로운 열로 추가
data['Action'] = actions

# 결과를 새로운 CSV 파일로 저장
data.to_csv('pqms_data_with_actions.csv', index=False)
