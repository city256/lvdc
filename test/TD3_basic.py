import pandas as pd
import datetime
from pytimekr import pytimekr
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
seed = 33
state_dim = 10  # 상태 벡터 차원
action_dim = 1  # 단일 action: 충/방전
hidden_dim = 128
max_action = 250
max_timesteps = 96
save_interval = 100
plot_interval = 100
reward_scaling_factor = 10  # 보상 정규화 스케일링

learning_rate = 1e-5 # 0.0003
gamma = 0.995
tau = 0.005
noise = 0.5
noise_clip = 0.8
noise_delay = 2
batch_size = 128
replay_buffer_size = 100000
episodes = 2000

window_size = 480

# 난수 시드 추가
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

train_csv = pd.read_csv('../test/train_data.csv')
d0907_csv = pd.read_csv('../test/0907_0911.csv')
workday_csv = pd.read_csv('workday.csv')
holiday_csv = pd.read_csv('holiday.csv')

# GPU 설정 (초기화 이후 어디서든 가능합니다)
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
    def __init__(self, window=window_size):
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
reward1_normalizer_ma = RewardNormalizer_ma()
reward2_normalizer_ma = RewardNormalizer_ma()

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
            #print('Done!!!')

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
        return reward / 100000

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
            print(f"DEBUG: NaN or inf in state: {normalized_state}, Step: {self.current_step}")

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
    # dt_15m_ago = dt - datetime.timedelta(minutes=15)
    month = dt.month
    hour = dt.hour
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

class TD3Agent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(TD3Agent, self).__init__()
        self.max_action = max_action

        # Actor 네트워크
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 출력이 [-1, 1] 범위로 제한됨
        )

        # Critic 네트워크 1
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Critic 네트워크 2
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.actor(state) * self.max_action

    def get_q_values(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.critic1(sa)
        q2 = self.critic2(sa)
        return q1, q2

class TD3:
    def __init__(self, state_dim, action_dim, hidden_dim=hidden_dim, max_action=max_action, gamma=gamma, tau=tau, lr=learning_rate, policy_noise=noise, noise_clip=noise_clip, policy_delay=noise_delay):
        self.actor = TD3Agent(state_dim, action_dim, hidden_dim).actor.to(device)
        self.actor_target = TD3Agent(state_dim, action_dim, hidden_dim).actor.to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = TD3Agent(state_dim, action_dim, hidden_dim).critic1.to(device)
        self.critic2 = TD3Agent(state_dim, action_dim, hidden_dim).critic2.to(device)
        self.critic1_target = TD3Agent(state_dim, action_dim, hidden_dim).critic1.to(device)
        self.critic2_target = TD3Agent(state_dim, action_dim, hidden_dim).critic2.to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0
        self.max_action = max_action

    def select_action(self, state, env, state_info, apply_constraints=True, noise=0.1):
        # 상태를 Tensor로 변환
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        # Actor 네트워크를 통해 액션 생성
        action = self.actor(state)

        # Tanh가 포함된 경우 max_action을 곱해 스케일링
        action = action * self.max_action  # Tanh 출력 스케일링

        if apply_constraints:
            # 현재 SoC 값 및 제약조건 고려
            current_soc = torch.tensor(state_info[0], dtype=torch.float32, device=device)
            next_load = torch.tensor(state_info[7], dtype=torch.float32, device=device)
            next_pv = torch.tensor(state_info[8], dtype=torch.float32, device=device)
            expected_grid = next_load - next_pv
            exp_grid_act = expected_grid + action / 4

            max_charge = min(env.conv_cap, (env.soc_max - current_soc.item()) * env.battery_cap)
            max_discharge = min(env.conv_cap, (current_soc.item() - env.soc_min) * env.battery_cap)

            # 역송 방지 및 SoC 제약조건 적용
            if exp_grid_act < 0:
                if current_soc < env.soc_max:
                    action = torch.clamp(-4 * exp_grid_act, max=max_charge)
                else:
                    action = torch.tensor([0.0], device=device)
            else:
                if action < 0:
                    max_discharge_to_prevent_reverse = min(-4 * (next_load - next_pv).item(), max_discharge)
                    action = torch.clamp(action, min=max_discharge_to_prevent_reverse)
                action = torch.clamp(action, min=-max_discharge, max=max_charge)

        if noise > 0:
            action += torch.randn_like(action) * noise  # Tensor에서 노이즈 추가

        # 최종적으로 conv_cap으로 제한
        action = torch.clamp(action, -env.conv_cap, env.conv_cap)

        # 환경과 상호작용을 위해 ndarray로 변환
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy

    def get_q_values(self, critic, state, action):
        # state와 action을 연결하여 Q 값 계산
        sa = torch.cat([state, action], dim=1)
        return critic(sa)

    def train(self, replay_buffer, batch_size=batch_size):
        self.total_it += 1

        # Replay Buffer에서 샘플링
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        # 데이터를 GPU로 이동
        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)
        reward = reward.to(device)
        done = done.to(device).unsqueeze(1)  # (batch_size, 1)로 차원 맞춤

        with torch.no_grad():
            # Target Policy Smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Q 값 계산
            target_q1 = self.get_q_values(self.critic1_target, next_state, next_action)  # (batch_size, 1)
            target_q2 = self.get_q_values(self.critic2_target, next_state, next_action)  # (batch_size, 1)

            # Reward와 Done의 차원 맞추기
            reward = reward.unsqueeze(-1) if reward.dim() == 1 else reward  # (batch_size, 1)
            done = done.unsqueeze(-1) if done.dim() == 1 else done  # (batch_size, 1)

            # Target Q 값 계산
            target_q = reward + (1 - done) * self.gamma * torch.min(target_q1, target_q2)


        # 현재 Q 값 계산
        current_q1 = self.get_q_values(self.critic1, state, action)
        current_q2 = self.get_q_values(self.critic2, state, action)

        # Critic Loss 계산 및 업데이트
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Policy Delay에 따라 Actor 업데이트
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.get_q_values(self.critic1, state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target 네트워크 업데이트
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class ReplayBuffer:
    def __init__(self, max_size=replay_buffer_size):
        self.storage = []
        self.max_size = int(max_size)  # 최대 크기
        self.ptr = 0  # 포인터 초기화

    def add(self, state, action, next_state, reward, done):
        """새로운 경험 추가"""
        transition = (state, action, next_state, reward, done)  # Transition 생성
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
    grid = [float(value.item()) if isinstance(value, (np.ndarray, torch.Tensor)) else float(value) for value in grid]
    load = [float(value.item()) if isinstance(value, (np.ndarray, torch.Tensor)) else float(value) for value in load]
    pv = [float(value.item()) if isinstance(value, (np.ndarray, torch.Tensor)) else float(value) for value in pv]
    action = [float(value.item()) if isinstance(value, (np.ndarray, torch.Tensor)) else float(value) for value in
              action]
    soc = [float(value.item()) if isinstance(value, (np.ndarray, torch.Tensor)) else float(value) for value in soc]

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

def train_td3(env, agent, replay_buffer, episodes=episodes, save_interval=save_interval, max_timesteps=max_timesteps, batch_size=batch_size):
    episode_rewards = []
    episode_total_costs = []
    best_total_cost = float('inf')  # 최소 total_cost를 추적
    best_model_path = "best_td3.pth"  # 최적 모델 저장 경로

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
            # 에이전트가 환경으로부터 액션을 선택
            action = agent.select_action(state, env, state_info, True)

            # 환경에서 한 스텝을 진행하여 다음 상태와 보상을 얻음
            next_state, reward, done, next_state_info = env.step(action)

            # 메모리에 현재 스텝 데이터 저장
            replay_buffer.add(state, action, next_state, reward, float(done))

            # Replay Buffer에 충분한 데이터가 있을 때 학습 수행
            if len(replay_buffer.storage) > batch_size:
                agent.train(replay_buffer, batch_size)

            # 데이터를 저장
            dates.append(env.date)
            load_values.append(state_info[1])
            pv_values.append(state_info[2])
            grid_values.append(state_info[3])
            soc_values.append(state_info[0])
            action_values.append(action)
            peak_values.append(env.peak_time)

            # 상태 업데이트
            state = next_state
            state_info = next_state_info
            total_reward += reward

            # 에피소드 종료 체크
            if done:
                break

        # 에피소드 보상 기록
        episode_rewards.append(total_reward)
        episode_total_costs.append(env.total_cost)

        # 최적 모델 저장
        if env.total_cost < best_total_cost:
            best_total_cost = env.total_cost
            torch.save(agent.actor.state_dict(), best_model_path)
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
            torch.save(agent.actor.state_dict(), f"td3_{episode + 1}_actor.pth")

    # 학습 보상 시각화
    episode_rewards.pop(0)
    episode_total_costs.pop(0)
    plt.figure(figsize=(12, 8))

    # 에피소드 보상 그래프
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward over Episodes(TD3)')
    plt.legend()

    # 에피소드 total_cost 그래프
    plt.subplot(2, 1, 2)
    plt.plot(episode_total_costs, label='Total Cost', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Cost')
    plt.title('Training Total Cost over Episodes(TD3)')
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
        f'Best TD3 {best_episode_num}'
    )

def run_test(env, model_path, agent, test=0, reference_value=0, scale=1):
    # 모델 로드
    agent.actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.actor.eval()  # Actor 네트워크를 평가 모드로 전환

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
        # 추론 시 탐색을 비활성화
        if test == 1:
            with torch.no_grad():
                action = agent.select_action(state, env, state_info, True, noise=0)
        else:
            with torch.no_grad():
                action = agent.select_action(state, env, state_info, False, noise=0.2)

        # Adjust action based on the reference value
        adjusted_action = action - reference_value

        next_state, reward, done, next_state_info = env.step(adjusted_action)  # adjusted_action 사용

        # 각 값들을 스칼라로 변환하여 저장
        soc_values.append(
            float(state_info[0][0]) if isinstance(state_info[0], np.ndarray) and np.ndim(state_info[0]) > 0 else float(state_info[0])
        )
        load_values.append(
            float(state_info[1][0]) if isinstance(state_info[1], np.ndarray) and np.ndim(state_info[1]) > 0 else float(state_info[1])
        )
        pv_values.append(
            float(state_info[2][0]) if isinstance(state_info[2], np.ndarray) and np.ndim(state_info[2]) > 0 else float(state_info[2])
        )
        grid_values.append(
            float(state_info[3][0]) if isinstance(state_info[3], np.ndarray) and np.ndim(state_info[3]) > 0 else float(state_info[3])
        )
        action_values.append(
            float(adjusted_action[0]) if isinstance(adjusted_action, np.ndarray) and np.ndim(adjusted_action) > 0 else float(adjusted_action)
        )
        peak_values.append(env.peak_time)
        date_values.append(env.date)

        total_reward += reward

        if done:
            break

        state = next_state
        state_info = next_state_info

    # 비용 정보 계산
    cost_info = {
        'total_cost': round(env.total_cost[0], -2),
        'usage_sum': round(env.usage_sum[0], -2),
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
    plot_results(date_values, load_values, pv_values, grid_values, action_values, soc_values, peak_values, model_path)
    return cost_info


test_csv = pd.read_csv('gloom_work.csv')
work_env = Environment(workday_csv)
hol_env = Environment(holiday_csv)
test_env = Environment(d0907_csv)
env = test_env

replay_buffer = ReplayBuffer()
td3_agent = TD3(state_dim, action_dim, hidden_dim)

# 학습 및 추론
train_td3(env, td3_agent, replay_buffer, episodes, max_timesteps=env.total_steps)
run_test(env, f"td3_{episodes}_actor.pth", td3_agent, 1, 0)
# run_test(env, f"td3_700_actor.pth", td3_agent, 1, 0)
