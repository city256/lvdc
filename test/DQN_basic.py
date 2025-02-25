import pandas as pd
import datetime
from pytimekr import pytimekr
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
# 제약조건, 전체요금 차
# hyperparameters
seed = 33
state_dim = 9  # 상태 벡터 차원
action_dim = 501  # 단일 action: 충/방전
hidden_dim = 128
learning_rate = 3e-4 # 0.0003
gamma = 0.995
eps_clip = 0.4
K_epochs = 4
max_action = 250
max_timesteps = 96
episodes = 1000
save_interval = 100
plot_interval = 100
batch_size = 96  # 미니배치 크기, batch_interval 보다 크면 안됨!!!
batch_interval = 192
reward_scaling_factor = 10  # 보상 정규화 스케일링


# 난수 시드 추가
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

train_csv = pd.read_csv('../test/train_data.csv')
d0907_csv = pd.read_csv('../test/0907_0911.csv')
workday_csv = pd.read_csv('workday.csv')
holiday_csv = pd.read_csv('holiday.csv')

# hyperparameter
w1 = 0.5 # cost weight
w2 = 0.4 # peak weight
w3 = 0.1 # swtich weight

class RewardNormalizer_zscore:
    def __init__(self):
        self.rewards = []

    def normalize(self, reward):
        reward = np.array(reward).ravel()
        self.rewards.append(reward)
        mean_reward = np.mean(self.rewards)
        std_reward = np.std(self.rewards) + 1e-5  # Prevent division by zero
        return (reward - mean_reward) / std_reward

class RewardNormalizer_ma:
    def __init__(self, window=100):
        self.rewards = []
        self.window = window

    def normalize(self, reward):
        reward = np.array(reward).ravel()
        self.rewards.append(reward)
        # 이동 평균 및 표준편차 계산 (window 크기만큼)
        if len(self.rewards) > self.window:
            self.rewards.pop(0)  # 윈도우를 유지하기 위해 오래된 값 제거

        mean_reward = np.mean(self.rewards)
        std_reward = np.std(self.rewards)

        # 표준편차가 0인 경우 예외 처리
        if std_reward < 1e-5:
            return reward - mean_reward  # 표준편차가 거의 0이면 단순히 평균만 빼서 반환

        return (reward - mean_reward) / (std_reward + 1e-5)

reward_normalizer_zs = RewardNormalizer_zscore()
reward_normalizer_ma = RewardNormalizer_ma()

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
        self.episode_reward = 0
        self.current_step = 0
        self.date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        self.demand_sum = 0

        # 모델링 변수 설정
        self.done = False
        self.soc = 0  # 초기 SoC 설정 (충전 상태)
        self.load = 0
        self.pv = 0
        self.price = 0
        self.peak_time = 0
        self.charge = 0
        self.discharge = 0
        self.ess_action = 0
        self.grid = 0
        self.peak = 0
        self.switch_sum = 0
        self.workday = 0
        self.usage_cost = 0
        self.usage_sum = 0
        self.total_cost = (self.peak * self.demand_cost + self.usage_sum) * 1.137
        self.previous_total_cost = self.total_cost  # 이전 스텝의 total_cost를 추적

        # 데이터에서 각 변수의 최대값과 최소값 계산
        self.max_load = self.data['load'].max()
        self.min_load = self.data['load'].min()
        self.max_pv = self.data['pv'].max()
        self.min_pv = self.data['pv'].min()
        self.max_grid = (self.data['load'] - self.data['pv']).max() + 250
        self.min_grid = (self.data['load'] - self.data['pv']).min() - 250
        self.max_peak = 500  # 피크는 grid의 최대값
        self.min_peak = 150  # 최소값은 0
        self.max_hour = 23
        self.min_hour = 0

    def reset(self):
        # 초기화 시 각 변수 리셋
        self.current_step = 0
        self.date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        self.month, self.day, self.hour = date(self.date)
        self.next_day_holiday = is_next_day_holiday(self.date)

        self.done = False
        self.soc = 0.5
        self.load = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'load'].iloc[0], 2))
        self.pv = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'pv'].iloc[0], 2))
        self.next_load =  float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step + 1, 'load'].iloc[0], 2))
        self.next_pv = float(round(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step+ 1 , 'pv'].iloc[0], 2))
        self.grid = self.load - self.pv
        self.price, self.peak_time = calculate_price(self.date)
        self.workday = check_workday(self.date)
        self.demand_sum = 0
        self.charge = 0
        self.discharge = 0
        self.ess_action = 0
        self.peak = calculate_peak(self.grid, self.load, self.pv, 0, self.contract)
        self.switch_sum = 0
        self.usage_cost = max(0, self.grid) * self.price
        self.usage_sum = self.usage_cost
        self.total_cost = (self.peak * self.demand_cost + self.usage_sum) * 1.137
        self.previous_total_cost = self.total_cost  # 초기화 시 이전 total_cost 설정
        self.cnd_state = [0]
        self.episode_reward = 0

        state = self.get_normalized_state()
        state_info = [self.soc, self.load, self.pv, self.grid, self.peak, self.peak_time, self.workday, self.next_load, self.next_pv, self.next_day_holiday]
        return state, state_info

    def step(self, action):
        if isinstance(action, torch.Tensor):  # ndarray
            action = action.item()  # 또는 action[0] 사용 가능 float, int
        # 현재 상태 업데이트
        current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        current_load = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'load'].iloc[0])
        current_pv = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'pv'].iloc[0])
        #print(f'[{self.current_step}]load/pv = {current_load}/{current_pv}')
        # 다음 스텝 정보 가져오기 (마지막 스텝이면 0으로 설정)
        if self.current_step < self.total_steps - 1:
            next_load = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step + 1, 'load'].iloc[0])
            next_pv = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step + 1, 'pv'].iloc[0])
        else:
            next_load = 0
            next_pv = 0
        current_grid = current_load - current_pv + (action / 4)
        current_soc = calculate_soc(self.soc, action, self.battery_cap)
        current_price, current_peak_time = calculate_price(self.date)
        # current_usage_cost = current_price * max(current_grid, 0)
        current_usage_cost = current_price * abs(current_grid)
        current_peak = calculate_peak(self.peak, current_load, current_pv, action, self.contract)

        self.month, self.day, self.hour = date(current_date)

        # 충방전량 저장
        if action > 0:
            current_charge = action / 4
            current_discharge = 0
        elif action < 0:
            current_charge = 0
            current_discharge = abs(action / 4)
        else:
            current_charge = 0
            current_discharge = 0

        # 변수 업데이트
        self.charge = current_charge
        self.discharge = current_discharge
        self.ess_action = action
        self.date = current_date
        self.load = current_load
        self.pv = current_pv
        self.next_load = next_load
        self.next_pv = next_pv
        self.grid = current_grid
        self.price = current_price
        self.peak_time = current_peak_time
        self.soc = current_soc
        self.workday = check_workday(self.date)
        self.next_day_holiday = is_next_day_holiday(self.date)
        self.peak = current_peak
        self.usage_cost = current_usage_cost
        self.usage_sum += current_usage_cost
        self.switch_sum += calculate_switch(action, self.cnd_state)
        self.demand_sum = self.peak * self.demand_cost
        self.total_cost = (self.demand_sum + self.usage_sum) * 1.137

        # 에피소드 종료 시 누적된 total_cost를 보상으로 반영
        if self.current_step >= self.total_steps - 1:
            self.done = True

        # 보상 계산
        reward = self.compute_reward(action)

        self.current_step += 1
        self.episode_reward += reward

        # 다음 상태에 적용
        next_state = self.get_normalized_state()
        next_state_info = [self.soc, self.load, self.pv, self.grid, self.peak, self.peak_time, self.workday, self.next_load, self.next_pv, self.next_day_holiday]
        #print(f'[{self.current_step-1}] : {next_state_info} = {action}')
        #print(f'[{self.current_step-1}] : {next_state} = {action}')
        #self.render(action, reward)
        return next_state, reward, self.done, next_state_info

    def compute_reward(self, action):
        current_total_cost = (self.peak * self.demand_cost + self.usage_sum) * 1.137
        reward = self.previous_total_cost - current_total_cost
        reward2 = -self.usage_cost

        # # 역송 방지 패널티
        # if self.grid < 0:  # 그리드가 음수일 때 (역송)
        #     reward -= 2.0 * abs(self.grid)  # 역송이 발생할 때 패널티 부여
        #     reward2 -= 2.0 * abs(self.grid)
        #
        #     # 추가 보상 요소
        # if self.peak_time == 0 and action > 0:  # 경부하 시간에 충전 시 보상
        #     reward += 0.2 * abs(action)
        #     reward2 += 0.2 * abs(action)
        # elif self.peak_time == 2 and action < 0:  # 최대부하 시간에 방전 시 보상
        #     reward += 0.3 * abs(action)
        #     reward2 += 0.3 * abs(action)
        #
        # # SoC 범위 유지 보상/패널티
        # if self.soc < self.soc_min:
        #     reward -= 1.0  # SoC가 soc_min 미만일 때 패널티
        #     reward2 -= 1.0
        # elif self.soc > self.soc_max:
        #     reward -= 1.0  # SoC가 soc_max 초과일 때 패널티
        #     reward2 -= 1.0

        self.previous_total_cost = current_total_cost
        normal_reward = reward_normalizer_ma.normalize(reward) / 10
        normal_reward2 = reward_normalizer_ma.normalize(reward2) / 10
        w = 0.5
        return reward / 100000
        # return (normal_reward * w) + (normal_reward2 * (1-w))

    def get_normalized_state(self):
        # SoC는 이미 0과 1 사이이므로 그대로 사용
        soc = float(self.soc.item()) if isinstance(self.soc, np.ndarray) else float(self.soc)
        load = float((self.load - self.min_load) / (self.max_load - self.min_load))
        pv = float((self.pv - self.min_pv) / (self.max_pv - self.min_pv))
        peak_value = (self.peak - self.min_peak) / (self.max_peak - self.min_peak)
        peak = float(peak_value.item()) if isinstance(peak_value, np.ndarray) and peak_value.size == 1 else float(peak_value)
        peak_time = float(self.peak_time / 2)  # peak_time은 0, 1, 2 중 하나이므로 0과 1 사이로 정규화
        workday = float(self.workday)  # 이미 0 또는 1이므로 그대로 사용
        hour = float(self.hour / 23)
        next_load = float((self.next_load - self.min_load) / (self.max_load - self.min_load))
        next_pv = float((self.next_pv - self.min_pv) / (self.max_pv - self.min_pv))

        normalized_state = [soc, load, pv, peak, peak_time, workday, hour, next_load, next_pv]

        if np.isnan(normalized_state).any() or np.isinf(normalized_state).any():
            print(f"[DEBUG] NaN or inf detected in state: {normalized_state}")
        return np.array(normalized_state)

    def render(self, action, reward):
        # 현재 상태를 출력
        print(f"Step: {self.current_step}")
        print(f"[{self.date}] SoC: {round(self.soc * 100, 2)}%")
        print(f"Load: {self.load}, PV: {self.pv}, Grid : {round(self.grid, 2)}, Price: {self.price}")
        print(f"Total: {round(self.total_cost)}, Demand: {round(self.demand_cost * self.peak)}, Usage: {round(self.usage_sum)}, Use: {round(self.usage_cost)}")
        print(f"Switch Sum: {self.switch_sum}, Peak: {self.peak}, Reward: {reward}")
        print(f'action: {action}')
        print("-" * 40)

def is_next_day_holiday(date_str):
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    holidays = pytimekr.holidays(date.year)  # 해당 연도의 공휴일 목록
    next_day = date + datetime.timedelta(days=1)
    is_holiday = next_day.date() in holidays  # 공휴일 여부
    is_weekend = next_day.weekday() >= 5  # 주말 여부 (5: 토요일, 6: 일요일)
    return 1 if is_holiday or is_weekend else 0  # 공휴일 또는 주말이면 1, 아니면 0
def date(date_str):
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

    # 필요한 정보 추출
    month = date_obj.month  # 1-12
    day = date_obj.weekday()  # 0=Monday, 6=Sunday
    hour = date_obj.hour  # 0-23

    # state에 추가
    return month, day, hour

def calculate_peak(grid_prev, load, pv, action, P_contract):
    max_P_grid = max(grid_prev, load - pv + action)  # grid의 최대값 계산

    if max_P_grid > P_contract * 0.3:
        return max_P_grid
    else:
        return P_contract * 0.3

def calculate_switch(ess, cnd_state):
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
    if 6 <= month <= 8:  # 여름철 (June-August)
        if 22 <= hour or hour <= 8:  # 경부하 (Off-peak)
            return 94, 0
        elif 8 <= hour <= 11 or 12 <= hour <= 13 or 18 <= hour <= 22:  # 중간부하 (Mid-peak)
            return 146.9, 1
        else:  # 최대부하 (Maximum peak)
            return 229, 2
    elif month in [11, 12, 1, 2]:  # 겨울철 (November-February)
        if 22 <= hour or hour <= 8:  # 경부하 (Off-peak)
            return 101, 0
        elif 8 <= hour <= 9 or 12 <= hour <= 16 or 19 <= hour <= 22:  # 중간부하 (Mid-peak)
            return 147.1, 1
        else:  # 최대부하 (Maximum peak)
            return 204.6, 2
    else:  # 봄, 가을철 (March-May, September-November)
        if 22 <= hour or hour <= 8:  # 경부하 (Off-peak)
            return 94, 0
        elif 8 <= hour <= 11 or 18 <= hour <= 22:  # 중간부하 (Mid-peak)
            return 116.5, 1
        elif 11 <= hour <= 12 or 13 <= hour <= 18:  # 최대부하 (Maximum peak)
            return 147.2, 2

def check_workday(date_str):

    date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:00')
    # 해당 날짜가 공휴일인지 확인
    holidays = pytimekr.holidays(date.year)

    if date.weekday() >= 5 or date.date() in holidays:
        return 0  # 공휴일 또는 주말
    else:
        return 1  # 평일, 주중

def calculate_soc(previous_soc, action, max_capacity):
    soc = ((max_capacity * previous_soc) + (action/4))/1000
    return max(0, min(1, soc))

# DQN 네트워크 정의
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# DQN 에이전트 클래스
class DQNAgent:
    def __init__(self, state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, lr=learning_rate, gamma=gamma, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.memory = []
        self.batch_size = batch_size
        self.lr = lr

        # 네트워크와 타겟 네트워크 생성
        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state, env):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 상태를 GPU로 이동

        # Epsilon-greedy 정책에 따라 행동 선택
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)  # 무작위 인덱스 선택
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_index = torch.argmax(q_values).item()  # 가장 큰 Q-값에 해당하는 행동 선택

        # 행동 인덱스를 실제 행동 값으로 변환 (-250 ~ 250)
        action = action_index - 250  # action_dim이 501이므로 -250에서 +250 범위의 정수

        # SoC 제약조건 적용
        current_soc = state[0, 0].item()  # 현재 SoC 값

        # 충전 및 방전 범위를 계산
        max_charge = float(min(env.conv_cap, (env.soc_max - current_soc) * env.battery_cap))
        max_discharge = float(min(env.conv_cap, (current_soc - env.soc_min) * env.battery_cap))

        # 현재 SoC와 action으로 예측된 SoC 계산
        predicted_soc = current_soc + (action / 4) / env.battery_cap

        # 방전량 조정: SoC가 `soc_min` 미만으로 내려가지 않도록 함
        if predicted_soc < env.soc_min:
            max_allowed_discharge = (current_soc - env.soc_min) * env.battery_cap * 4  # 방전량을 조정
            action = max(action, -max_allowed_discharge)

        # 충전량 조정: SoC가 `soc_max`를 초과하지 않도록 함
        if predicted_soc > env.soc_max:
            max_allowed_charge = (env.soc_max - current_soc) * env.battery_cap * 4  # 충전량을 조정
            action = min(action, max_allowed_charge)

        # 최종 SoC 범위 내에서 행동 제한
        action = np.clip(action, -max_discharge, max_charge)

        # print(f"Current SoC: {current_soc:.2f}, Predicted SoC: {predicted_soc:.2f}, Action: {action}")

        return action  # 제한된 행동 반환

    def store_transition(self, transition):
        self.memory.append(transition)
        if len(self.memory) > 10000:  # 최대 메모리 크기 제한
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # 미니배치 샘플링
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 배열 변환 최적화 및 차원 맞추기
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1).to(device)  # 차원 조정
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1).to(device)  # 차원 조정

        # actions 값이 action_dim 내의 양수 인덱스로 변환되도록 보장
        actions = actions + (self.action_dim // 2)  # 음수 인덱스를 양수 인덱스로 변환

        # 현재 상태에 대한 Q값 계산
        current_q_values = self.policy_net(states).gather(1, actions)

        # 다음 상태에서의 최대 Q값 계산
        next_q_values = self.target_net(next_states).max(1)[0].detach().view(-1, 1)  # 차원 조정
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # target_q_values의 크기가 current_q_values와 맞는지 확인
        if target_q_values.size() != current_q_values.size():
            print(
                f"Debug: current_q_values size: {current_q_values.size()}, target_q_values size: {target_q_values.size()}")
            target_q_values = target_q_values[:current_q_values.size(0)]  # 크기 맞춤

        # 손실 함수 계산 및 역전파
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)  # Huber Loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 탐욕 정책 업데이트
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update_target_network(self, tau=0.005):
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def plot_results(dates, load, pv, grid, action, soc, e_num):
    # 모든 리스트가 동일한 길이인지 확인
    assert len(dates) == len(load) == len(pv) == len(grid) == len(action) == len(soc), "리스트 길이가 일치하지 않습니다."

    # 각 리스트가 스칼라 값으로만 이루어져 있는지 검사
    for i, (d, l, p, g, a, s) in enumerate(zip(dates, load, pv, grid, action, soc)):
        assert np.isscalar(d) and np.isscalar(l) and np.isscalar(p) and np.isscalar(g) and np.isscalar(
            a) and np.isscalar(s), f"리스트 {i}번째 요소가 스칼라가 아닙니다."

    plt.figure(figsize=(14, 8))

    # 첫 번째 그래프: Load, PV, Grid (왼쪽 y축), SoC (오른쪽 y축)
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(dates, load, label='Load', color='blue', linewidth=2)
    ax1.plot(dates, pv, label='PV', color='orange', linewidth=2)
    ax1.plot(dates, grid, label='Grid', color='green', linewidth=2)
    ax1.set_ylabel('Load / PV / Grid')
    ax1.legend(loc='upper left')

    # SoC 값을 첫 번째 그래프의 오른쪽 y축에 표시, 범위 고정
    ax2 = ax1.twinx()
    ax2.plot(dates, soc, label='SoC', color='purple', linestyle='--', linewidth=2)
    ax2.set_ylabel('SoC')
    ax2.set_ylim(0, 1)  # SoC의 y축 범위를 0에서 1로 고정
    ax2.legend(loc='upper right')

    # 두 번째 그래프: Action을 점으로만 표시하고 SoC (왼쪽 y축에 Action, 오른쪽 y축에 SoC 범위 표시)
    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(dates, action, label='Action', color='red', linestyle='None', marker='o')  # 점으로만 표시
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Action')
    ax3.legend(loc='upper left')

    # SoC 값을 오른쪽 y축에 표시, 범위 고정
    ax4 = ax3.twinx()
    ax4.plot(dates, soc, label='SoC', color='purple', linestyle='--', linewidth=2)
    ax4.set_ylabel('SoC')
    ax4.set_ylim(0, 1)  # SoC의 y축 범위를 0에서 1로 고정
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def train_dqn(env, agent, episodes=episodes, save_interval=save_interval, max_timesteps=max_timesteps, batch_interval = batch_interval):
    timestep = 0
    episode_rewards = []
    actions = []

    for episode in range(episodes):
        state, state_info = env.reset()
        total_reward = 0

        for t in range(max_timesteps):
            # 에이전트가 환경으로부터 액션을 선택
            action = agent.select_action(state, env)
            # print(f'[{env.current_step}] {env.soc} : {action - agent.action_dim // 2}')

            # 환경에서 한 스텝을 진행하여 다음 상태와 보상을 얻음
            next_state, reward, done, _ = env.step(action)  # action_dim의 절반을 빼서 음수/양수 변환

            agent.store_transition((state, action, reward, next_state, done))
            agent.train()

            state = next_state
            total_reward += reward
            timestep += 1

            if done:
                break

        # 타겟 네트워크 주기적으로 업데이트
        if (episode + 1) % batch_interval == 0:
            agent.update_target_network()


        # 에피소드 보상 기록
        episode_rewards.append(total_reward)

        # 에피소드 정보 출력
        print(
            f'Episode {episode + 1}/{episodes} finished with reward: {total_reward.item() if isinstance(total_reward, np.ndarray) else float(total_reward):.2f} / {env.total_cost}')
        # 모델 저장
        if (episode + 1) % save_interval == 0:
            torch.save(agent.policy_net.state_dict(), f"dqn_{episode + 1}.pth")

    # 학습 보상 시각화
    episode_rewards.pop(0)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward over Episodes(DQN)')
    plt.show()

def run_test(env, model_path, agent, reference_value=0, scale=1):
    # 모델 로드
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.policy_net.eval()  # 평가 모드로 설정

    state, state_info = env.reset()

    # 결과 저장을 위한 리스트 초기화
    pv_values = []
    load_values = []
    grid_values = []
    soc_values = []
    action_values = []
    date_values = []

    total_reward = 0

    # 테스트 루프
    for t in range(env.total_steps):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = agent.policy_net(state_tensor)
            action_index = torch.argmax(q_values).item()  # 최대 Q-값에 해당하는 행동 선택
            action = action_index - (agent.action_dim // 2)  # 음수/양수 범위로 변환

        # SoC 제약조건 적용
        current_soc = env.soc  # 현재 SoC 값

        # 충전 및 방전 범위를 계산
        max_charge = float(min(env.conv_cap, (env.soc_max - current_soc) * env.battery_cap))
        max_discharge = float(min(env.conv_cap, (current_soc - env.soc_min) * env.battery_cap))

        # 현재 SoC와 action으로 예측된 SoC 계산
        predicted_soc = current_soc + (action / 4) / env.battery_cap

        # 방전량 조정: SoC가 `soc_min` 미만으로 내려가지 않도록 함
        if predicted_soc < env.soc_min:
            max_allowed_discharge = (current_soc - env.soc_min) * env.battery_cap * 4  # 방전량을 조정
            action = max(action, -max_allowed_discharge)

        # 충전량 조정: SoC가 `soc_max`를 초과하지 않도록 함
        if predicted_soc > env.soc_max:
            max_allowed_charge = (env.soc_max - current_soc) * env.battery_cap * 4  # 충전량을 조정
            action = min(action, max_allowed_charge)

        # 최종 SoC 범위 내에서 행동 제한
        action = np.clip(action, -max_discharge, max_charge)

        # 행동 조정 (예: 기준값을 뺀다거나 하는 방식)
        adjusted_action = action - reference_value

        # 환경에서 스텝 진행
        next_state, reward, done, next_state_info = env.step(adjusted_action)

        # 각 값들을 스칼라로 변환하여 저장
        pv_values.append(float(state_info[2]))    # 발전 값
        load_values.append(float(state_info[1]))  # 부하 값
        grid_values.append(float(state_info[3]))  # 그리드 값
        soc_values.append(float(state_info[0]))   # 충전 상태 값
        action_values.append(float(adjusted_action))  # 조정된 action 저장
        date_values.append(env.date)

        total_reward += reward

        if done:
            break

        # 다음 상태로 전환
        state = next_state
        state_info = next_state_info

    # 비용 정보 계산
    cost_info = {
        'total_cost': env.total_cost,
        'usage_sum': env.usage_sum,
        'demand': env.demand_cost * env.peak,
        'switch_sum': env.switch_sum,
        'final_reward': total_reward
    }

    # 비용 정보 출력
    print(f"\nCost information from test run:")
    print(f"Total cost: {cost_info['total_cost']}")
    print(f"Usage sum: {cost_info['usage_sum']}")
    print(f"Demand cost: {cost_info['demand']}")
    print(f"Switch count: {cost_info['switch_sum']}")
    print(f"Final reward: {cost_info['final_reward']}")

    # 테스트 결과 시각화
    plot_results(date_values, load_values, pv_values, grid_values, action_values, soc_values, 0)

    return cost_info


test_csv = pd.read_csv('gloom_hol.csv')
work_env = Environment(workday_csv)
hol_env = Environment(holiday_csv)
test_env = Environment(test_csv)
env = test_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn_agent = DQNAgent(state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

# 학습 및 추론
train_dqn(env, dqn_agent, max_timesteps=env.total_steps)
run_test(env, f"dqn_{episodes}.pth", dqn_agent)
# run_test(env, f"dqn_400.pth", dqn_agent, 0,1)