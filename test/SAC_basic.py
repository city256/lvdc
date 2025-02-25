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
state_dim = 10  # 상태 벡터 차원
action_dim = 1  # 단일 action: 충/방전
max_action = 250
max_timesteps = 96
save_interval = 100
plot_interval = 100
reward_scaling_factor = 10  # 보상 정규화 스케일링

learning_rate = 1e-4
gamma = 0.995
tau = 0.005
alpha = 0.7
hidden_dim = 256
batch_size = 128
replay_buffer_size = 100000
episodes = 2000

# 난수 시드 추가
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

train_csv = pd.read_csv('../test/train_data.csv')
d0907_csv = pd.read_csv('../test/0907_0911.csv')
workday_csv = pd.read_csv('workday.csv')
holiday_csv = pd.read_csv('holiday.csv')

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, window=500):
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

        self.previous_total_cost = current_total_cost
        normal_reward = reward_normalizer_ma.normalize(reward) / 10
        normal_reward2 = reward_normalizer_ma.normalize(reward2) / 10
        w = 0.6
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
        next_day = float(self.next_day_holiday)

        normalized_state = [soc, load, pv, peak, peak_time, workday, hour, next_load, next_pv, next_day]

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

class SACAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=hidden_dim, max_action=max_action):
        super(SACAgent, self).__init__()
        self.max_action = max_action

        # Example: Layer Normalization 추가
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # Log standard deviation for Gaussian policy
        # SACAgent 클래스에서 log_std 초기값 수정
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)  # 초기 표준편차 증가


        # Critic 네트워크 1
        # Critic 네트워크에 LayerNorm 추가
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Critic 네트워크 2
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action=None):
        mean = self.actor(state)
        std = torch.exp(self.log_std.clamp(-20, 2))  # 클리핑 범위 확인
        dist = Normal(mean, std)

        if action is not None:
            sa = torch.cat([state, action], dim=-1)
            q1 = self.critic1(sa)
            q2 = self.critic2(sa)
            return dist, q1, q2

        return dist

    def select_action(self, state, env, state_info, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = self.forward(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        return action.clamp(-self.max_action, self.max_action).detach().cpu().numpy().flatten()


class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim=hidden_dim, max_action=max_action, gamma=gamma, tau=tau, lr=learning_rate, alpha=0.2):
        self.actor_critic = SACAgent(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(
            list(self.actor_critic.critic1.parameters()) + list(self.actor_critic.critic2.parameters()), lr=lr
        )
        self.target_critic1 = SACAgent(state_dim, action_dim, hidden_dim, max_action).critic1.to(device)
        self.target_critic2 = SACAgent(state_dim, action_dim, hidden_dim, max_action).critic2.to(device)
        self.target_critic1.load_state_dict(self.actor_critic.critic1.state_dict())
        self.target_critic2.load_state_dict(self.actor_critic.critic2.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

        # Automatic alpha adjustment
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        self.alpha = alpha
        # Target Entropy 설정 강화
        self.target_entropy = -0.5 * float(action_dim)  # 탐색 성향 증가
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def update(self, replay_buffer, batch_size=batch_size):
        # 샘플링
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        # Critic 업데이트
        with torch.no_grad():
            next_action_dist = self.actor_critic(next_state)
            next_action = next_action_dist.rsample()
            log_prob = next_action_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            target_q1 = self.target_critic1(torch.cat([next_state, next_action], dim=-1))
            target_q2 = self.target_critic2(torch.cat([next_state, next_action], dim=-1))
            target_q = reward + self.gamma * (1 - done) * (torch.min(target_q1, target_q2) - self.alpha * log_prob)

        current_q1 = self.actor_critic.critic1(torch.cat([state, action], dim=-1))
        current_q2 = self.actor_critic.critic2(torch.cat([state, action], dim=-1))
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 업데이트
        action_dist = self.actor_critic(state)
        new_action = action_dist.rsample()
        log_prob = action_dist.log_prob(new_action).sum(dim=-1, keepdim=True)
        q1 = self.actor_critic.critic1(torch.cat([state, new_action], dim=-1))
        q2 = self.actor_critic.critic2(torch.cat([state, new_action], dim=-1))
        actor_loss = (self.alpha * log_prob - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha 업데이트 (자동 조정)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()  # Ensure log_alpha requires grad
        self.alpha_optimizer.step()

        # Alpha 업데이트
        self.alpha = self.log_alpha.exp().item()

        # Target Critic 업데이트
        with torch.no_grad():
            for param, target_param in zip(self.actor_critic.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_critic.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, max_size=replay_buffer_size):
        self.storage = []
        self.max_size = int(max_size)  # 최대 크기
        self.ptr = 0  # 포인터 초기화

    def add(self, state, action, next_state, reward, done):
        # Ensure all inputs are Tensors
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state)
        if not isinstance(reward, torch.Tensor):
            reward = torch.FloatTensor(reward)  # Ensure reward is a single value tensor
        if not isinstance(done, torch.Tensor):
            done = torch.FloatTensor([done])  # Ensure done is a single value tensor

        transition = (state, action, next_state, reward, done)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        """버퍼에서 랜덤으로 샘플링"""
        if batch_size > len(self.storage):
            raise ValueError(f"Requested batch_size {batch_size} exceeds buffer size {len(self.storage)}.")

        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state, action, next_state, reward, done = [], [], [], [], []

        for i in ind:
            s, a, ns, r, d = self.storage[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            next_state.append(np.array(ns, copy=False))
            reward.append(np.array(r, copy=False))
            done.append(np.array(d, copy=False))

        return (
            torch.FloatTensor(np.array(state)).to(device),  # (batch_size, state_dim)
            torch.FloatTensor(np.array(action)).to(device),  # (batch_size, action_dim)
            torch.FloatTensor(np.array(next_state)).to(device),  # (batch_size, state_dim)
            torch.FloatTensor(np.array(reward)).to(device),  # (batch_size, 1)
            torch.FloatTensor(np.array(done)).to(device)  # (batch_size, 1)
        )

def plot_results(dates, load, pv, grid, action, soc, peak_times, title, step_interval=95):
    # 데이터를 스칼라 값으로 변환 (평탄화)
    grid = [float(value.item()) if isinstance(value, np.ndarray) and value.size == 1 else float(value) for value in
            grid]
    load = [float(value.item()) if isinstance(value, np.ndarray) and value.size == 1 else float(value) for value in
            load]
    pv = [float(value.item()) if isinstance(value, np.ndarray) and value.size == 1 else float(value) for value in pv]
    action = [float(value.item()) if isinstance(value, np.ndarray) and value.size == 1 else float(value) for value in
              action]
    soc = [float(value.item()) if isinstance(value, np.ndarray) and value.size == 1 else float(value) for value in soc]

    # 모든 데이터가 동일한 길이인지 확인
    assert len(dates) == len(load) == len(pv) == len(grid) == len(action) == len(soc), "데이터 길이가 일치하지 않습니다."

    plt.figure(figsize=(14, 8))

    # 첫 번째 그래프: Load, PV, Grid (왼쪽 y축), SoC (오른쪽 y축)
    ax1 = plt.subplot(2, 1, 1)
    plt.title(f'{title} Episodes')

    # Peak time 배경색 추가
    for i in range(len(peak_times)):
        if peak_times[i] == 0:  # Off-peak
            ax1.axvspan(i, i + 1, facecolor='lightblue', alpha=0.3)
        elif peak_times[i] == 1:  # Mid-peak
            ax1.axvspan(i, i + 1, facecolor='lightgreen', alpha=0.3)
        elif peak_times[i] == 2:  # Peak
            ax1.axvspan(i, i + 1, facecolor='lightcoral', alpha=0.3)

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

    ax1.set_xticks(range(0, len(dates), step_interval))  # 특정 간격마다 tick 설정
    ax1.set_xticklabels(range(1, len(dates)+1, step_interval), rotation=45)  # 각 tick에 대해 텍스트 추가

    # 두 번째 그래프: Action을 점으로만 표시하고 SoC (왼쪽 y축에 Action, 오른쪽 y축에 SoC 범위 표시)
    ax3 = plt.subplot(2, 1, 2)

    # Peak time 배경색 추가
    for i in range(len(peak_times)):
        if peak_times[i] == 0:  # Off-peak
            ax3.axvspan(i, i + 1, facecolor='lightblue', alpha=0.3)
        elif peak_times[i] == 1:  # Mid-peak
            ax3.axvspan(i, i + 1, facecolor='lightgreen', alpha=0.3)
        elif peak_times[i] == 2:  # Peak
            ax3.axvspan(i, i + 1, facecolor='lightcoral', alpha=0.3)

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

    # x축에 간격 있는 단위 설정
    ax3.set_xticks(range(0, len(dates), step_interval))  # 특정 간격마다 tick 설정
    ax3.set_xticklabels(range(1, len(dates)+1, step_interval), rotation=45)  # 각 tick에 대해 텍스트 추가

    plt.tight_layout()
    plt.show()

def train_sac(env, agent, replay_buffer, episodes=episodes, save_interval=save_interval, max_timesteps=max_timesteps, batch_size=batch_size):
    episode_rewards = []
    episode_total_costs = []
    best_total_cost = float('inf')  # 최소 total_cost를 추적
    best_model_path = "best_sac.pth"  # 최적 모델 저장 경로

    best_episode_data = {
        "dates": [],
        "load": [],
        "pv": [],
        "grid": [],
        "action": [],
        "soc": [],
        "peak_time": []
    }

    for episode in range(episodes):
        state, state_info = env.reset()
        total_reward = 0

        # 에피소드 데이터를 저장할 리스트 초기화
        dates = []
        load_values = []
        pv_values = []
        grid_values = []
        soc_values = []
        action_values = []
        peak_values = []

        for t in range(max_timesteps):
            action = agent.actor_critic.select_action(state, env, state_info)
            next_state, reward, done, next_state_info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, float(done))

            if len(replay_buffer.storage) > batch_size:
                agent.update(replay_buffer)

            # 데이터를 저장
            dates.append(env.date)
            load_values.append(state_info[1])
            pv_values.append(state_info[2])
            grid_values.append(state_info[3])
            soc_values.append(state_info[0])
            action_values.append(action)
            peak_values.append(env.peak_time)

            state = next_state
            state_info = next_state_info
            total_reward += reward

            if done:
                break

        # 에피소드 보상 기록
        episode_rewards.append(total_reward)
        episode_total_costs.append(env.total_cost)

        if env.total_cost < best_total_cost:
            best_total_cost = env.total_cost
            torch.save(agent.actor_critic.actor.state_dict(), best_model_path)
            print(f"New best model saved at Episode {episode + 1} with total_cost: {env.total_cost}")

            # 최적 모델의 데이터를 저장
            best_episode_data["dates"] = dates
            best_episode_data["load"] = load_values
            best_episode_data["pv"] = pv_values
            best_episode_data["grid"] = grid_values
            best_episode_data["action"] = action_values
            best_episode_data["soc"] = soc_values
            best_episode_data["peak_time"] = peak_values
            best_episode_num = episode

        # 에피소드 정보 출력
        print(
            f'Episode {episode + 1}/{episodes} finished with reward: {total_reward} / {env.total_cost}'
        )

        # 일정 에피소드마다 모델 저장
        if (episode + 1) % save_interval == 0:
            torch.save(agent.actor_critic.actor.state_dict(), f"sac_{episode + 1}_actor.pth")

    # 학습 보상 시각화
    episode_rewards.pop(0)
    episode_total_costs.pop(0)
    plt.figure(figsize=(12, 8))

    # 에피소드 보상 그래프
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward over Episodes(SAC)')
    plt.legend()

    # 에피소드 total_cost 그래프
    plt.subplot(2, 1, 2)
    plt.plot(episode_total_costs, label='Total Cost', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Cost')
    plt.title('Training Total Cost over Episodes(SAC)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 최적 모델의 결과 시각화
    print(f"Best model's total cost: {best_total_cost}")
    plot_results(
        best_episode_data["dates"],
        best_episode_data["load"],
        best_episode_data["pv"],
        best_episode_data["grid"],
        best_episode_data["action"],
        best_episode_data["soc"],
        best_episode_data["peak_time"],
        f'Best SAC {best_episode_num}'
    )

def run_test(env, model_path, agent, test=0, reference_value=0, scale = 1):
    # 모델 로드
    agent.actor_critic.actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.actor_critic.actor.eval()

    state, state_info = env.reset()

    # 결과 저장을 위한 리스트 초기화
    pv_values = []
    load_values = []
    grid_values = []
    soc_values = []
    action_values = []
    date_values = []
    peak_values = []

    total_reward = 0

    # 테스트 루프
    for t in range(env.total_steps):
        # 추론 시에는 탐색을 비활성화합니다

        if test == 1:
            with torch.no_grad():
                action = agent.actor_critic.select_action(state, env, state_info, deterministic=True)
        else:
            with torch.no_grad():
                action = agent.actor_critic.select_action(state, env, state_info, deterministic=False)

        # Adjust action based on the reference value
        adjusted_action = action - reference_value

        next_state, reward, done, next_state_info = env.step(adjusted_action)  # adjusted action 사용

        # 각 값들을 스칼라로 변환하여 저장
        soc_values.append(float(state_info[0].item() if isinstance(state_info[0], np.ndarray) else state_info[0]))
        load_values.append(float(state_info[1].item() if isinstance(state_info[1], np.ndarray) else state_info[1]))
        pv_values.append(float(state_info[2].item() if isinstance(state_info[2], np.ndarray) else state_info[2]))
        grid_values.append(float(state_info[3].item() if isinstance(state_info[3], np.ndarray) else state_info[3]))
        action_values.append(float(adjusted_action.item() if isinstance(adjusted_action, np.ndarray) else adjusted_action))
        peak_values.append(env.peak_time)
        date_values.append(env.date)

        total_reward += reward

        if done:
            break

        state = next_state
        state_info = next_state_info

    # 비용 정보 계산
    cost_info = {
        'total_cost': round(env.total_cost[0,0], -2),
        'usage_sum': round(env.usage_sum[0,0], -2),
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

    print(f'[{env.current_step}] T{env.date} == {env.peak_time}')

    # 테스트 결과 시각화
    plot_results(date_values, load_values, pv_values, grid_values, action_values, soc_values, peak_values, model_path)
    return cost_info

test_csv = pd.read_csv('gloom_work.csv')
work_env = Environment(workday_csv)
hol_env = Environment(holiday_csv)
test_env = Environment(d0907_csv)
env = test_env

replay_buffer = ReplayBuffer()
sac_agent = SAC(state_dim, action_dim, hidden_dim)

# 메모리 및 에이전트 초기화
train_sac(env, sac_agent, replay_buffer, episodes, max_timesteps=env.total_steps)
# run_test(env, f"custom_{episodes}.pth", sac_agent, 1, 0)
# run_test(env, f"custom_100.pth", sac_agent, 1, 0, 1)