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
action_dim = 1  # 단일 action: 충/방전
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
        state_info = [self.soc, self.load, self.pv, self.grid, self.peak, self.peak_time, self.workday, self.month, self.day, self.hour, self.next_load, self.next_pv]
        return state, state_info

    def step(self, action):
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
        #current_usage_cost = current_price * max(current_grid, 0)
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
        self.peak = current_peak
        self.usage_cost = current_usage_cost
        self.usage_sum += current_usage_cost
        self.switch_sum += calculate_switch(action, self.cnd_state)
        self.demand_sum = self.peak * self.demand_cost
        self.total_cost = (self.demand_sum + self.usage_sum) * 1.137

        # 보상 계산
        reward = self.compute_reward(action)

        # 에피소드 종료 시 누적된 total_cost를 보상으로 반영
        if self.current_step >= self.total_steps - 1:
            self.done = True

        self.current_step += 1
        self.episode_reward += reward

        # 다음 상태에 적용
        next_state = self.get_normalized_state()
        next_state_info = [self.soc, self.load, self.pv, self.grid, self.peak, self.peak_time, self.workday, self.month, self.day, self.hour, self.next_load, self.next_pv]
        #print(f'[{self.current_step-1}] : {next_state_info} = {action}')
        #print(f'[{self.current_step-1}] : {next_state} = {action}')
        #self.render(action, reward)
        return next_state, reward, self.done, next_state_info

    def compute_reward(self, action):
        current_total_cost = (self.peak * self.demand_cost + self.usage_sum) * 1.137
        reward = self.previous_total_cost - current_total_cost
        reward2 = -self.usage_cost

        self.previous_total_cost = current_total_cost
        normal_reward = reward_normalizer_ma.normalize(reward) / 10
        normal_reward2 = reward_normalizer_ma.normalize(reward2) / 10
        w = 0.5
        # return normal_reward
        return (normal_reward * w) + (normal_reward2 * (w-1))

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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 행동 범위가 -1 ~ 1로 제한됨
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class TD3:
    def __init__(self, state_dim, action_dim, hidden_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_delay=2):
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, memory, batch_size=100):
        self.total_it += 1

        # 샘플링
        state, action, next_state, reward, done = memory.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)

        # Target policy noise 추가 및 클리핑
        noise = torch.FloatTensor(action).data.normal_(0, self.policy_noise).to(device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)

        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Target Q-value 계산
        target_q1, target_q2 = self.critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + ((1 - done) * self.gamma * target_q).detach()

        # Critic 업데이트
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Policy 업데이트 (지연된 업데이트)
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target 네트워크 업데이트
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, state, action, next_state, reward, done):
        data = (state, action, next_state, reward, done)

        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state, action, next_state, reward, done = zip(*[self.storage[i] for i in ind])

        return (
            np.array(state),
            np.array(action),
            np.array(next_state),
            np.array(reward).reshape(-1, 1),
            np.array(done).reshape(-1, 1),
        )


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


def train_ppo(env, agent, memory, episodes=episodes, save_interval=save_interval, max_timesteps=max_timesteps, batch_interval = batch_interval):
    timestep = 0
    episode_rewards = []
    actions = []

    for episode in range(episodes):
        state, state_info = env.reset()
        total_reward = 0

        for t in range(max_timesteps):
            # 에이전트가 환경으로부터 액션을 선택
            action = agent.select_action(state, memory, env)

            # 환경에서 한 스텝을 진행하여 다음 상태와 보상을 얻음
            next_state, reward, done, next_state_info = env.step(action)

            # 메모리에 보상과 종료 여부를 저장
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            total_reward += reward

            # 누적 보상 계산
            #print(f'[{env.current_step}] reward = {reward}, total = {total_reward} / {env.total_cost}')

            # 상태를 업데이트
            state = next_state
            timestep += 1

            # batch_interval마다 업데이트 및 메모리 초기화
            if timestep % batch_interval == 0:
                agent.update(memory)
                memory.clear_memory()

            # 에피소드 종료 체크
            if done:
                break

        # 메모리에 저장된 action들 자료형 ndarray로 통일 ==> select_action에서 변경시 안해도 됨
        # memory.actions = [np.array([[action]]) if not isinstance(action, np.ndarray) else action for action in
        #                   memory.actions]

        # 에피소드가 끝날 때마다 정책 업데이트
        if timestep % batch_interval != 0:
            agent.update(memory)
            memory.clear_memory()

        # agent.update(memory)
        # memory.clear_memory()

        # 에피소드 보상 기록
        episode_rewards.append(total_reward)

        # 에피소드 정보 출력
        print(
            f'Episode {episode + 1}/{episodes} finished with reward: {total_reward.item() if isinstance(total_reward, np.ndarray) else float(total_reward):.2f} / {env.total_cost}')
        # 모델 저장
        if (episode + 1) % save_interval == 0:
            torch.save(agent.policy.state_dict(), f"basic_{episode + 1}.pth")

    # 학습 보상 시각화
    episode_rewards.pop(0)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward over Episodes(Basic)')
    plt.show()

def run_test(env, model_path, agent, test=0, reference_value=0, scale = 1):
    # 모델 로드
    agent.policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.policy.eval()

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
        # 추론 시에는 탐색을 비활성화합니다
        if test == 1:
            with torch.no_grad():
                action, _ = agent.policy.test_act(torch.FloatTensor(state).unsqueeze(0), env, scale)
        else:
            with torch.no_grad():
                action, _ = agent.policy.act(torch.FloatTensor(state).unsqueeze(0))

        # Adjust action based on the reference value
        adjusted_action = action[0].item() - reference_value

        next_state, reward, done, next_state_info = env.step(adjusted_action)  # adjusted action 사용

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
env = work_env

# 메모리 및 에이전트 초기화
memory = Memory()
ppo_agent = PPO(state_dim, action_dim, hidden_dim=hidden_dim)

# GPU 설정 (초기화 이후 어디서든 가능합니다)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ppo_agent.policy.to(device)  # 모델을 GPU로 이동

# 학습 및 추론
train_ppo(env, ppo_agent, memory, max_timesteps=env.total_steps)
run_test(env, f"basic_{episodes}.pth", ppo_agent)
# run_test(env, f"basic_600.pth", ppo_agent, 1,170, 1.4)