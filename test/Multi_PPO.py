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
hidden_dim = 256
max_action = 250
max_timesteps = 96
save_interval = 100
plot_interval = 100
reward_scaling_factor = 10  # 보상 정규화 스케일링

learning_rate = 1e-4
gamma = 0.995
eps_clip = 0.6
K_epochs = 4
batch_num = 96
batch_interval = 192
episodes = 3000

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
            return reward - mean_reward   # 표준편차가 거의 0이면 단순히 평균만 빼서 반환

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

        # 보상 계산
        reward = self.compute_reward(action)

        # 에피소드 종료 시 누적된 total_cost를 보상으로 반영
        if self.current_step >= self.total_steps - 1:
            self.done = True
            #print('Done!!!')

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

        # if 0 <= self.load - self.pv + (action/4) <= 10 :
        #     reward = abs(reward)
        #     reward2 = abs(reward2)
        #
        # elif self.next_load - self.next_pv < 0 :
        #     reward = reward * 2
        #
        # if self.soc_min <= self.soc <= self.soc_max:
        #     reward *= 0.9
        # else:
        #     reward *= 1.1
        #
        # # 주말 역송시 충전 보상
        # if self.next_load - self.next_pv < 0 and abs(self.next_load - self.next_pv)*0.7 < action/4 <= abs(self.next_load - self.next_pv) :
        #     reward += abs(self.next_load - self.next_pv + (action/4)) * self.price * 10
        #     reward2 += abs(self.next_load - self.next_pv + (action/4)) * self.price * 10
        #     # print(self.next_load - self.next_pv, reward, action, 'Good Action')

        self.previous_total_cost = current_total_cost
        normal_reward = reward_normalizer_ma.normalize(reward) / 10
        normal_reward2 = reward_normalizer_ma.normalize(reward2) / 10
        w = 0.6
        # return normal_reward2
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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=250):
        super(ActorCritic, self).__init__()

        self.max_action = max_action

        # Actor actor with additional layers and larger hidden dimensions
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic actor with a similar deeper structure
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Additional log standard deviation layer for the Gaussian policy
        self.log_std = nn.Parameter(torch.full((1, action_dim), 1.8))  # 초기 표준편차를 더 크게 설정

    def forward(self, state):
        mean = self.actor(state)
        # NaN 검사 추가
        if torch.isnan(mean).any():
            print("NaN detected in mean after actor actor")
            print(f"Mean: {mean}")
            print(f"State: {state}")
        std = self.log_std.exp().expand_as(mean)
        std = torch.clamp(std, min=1e-5)
        dist = Normal(mean, std)
        value = self.critic(state)
        return dist, value

    def test_act(self, state, env, state_info, scale=1):
        dist, _ = self.forward(state)
        action = dist.sample()

        # 현재 SoC 값 가져오기
        current_soc = state[0, 0].item()
        # next_load와 next_pv를 가져오기 (정규화된 값을 비정규화)
        next_load = state_info[7]
        next_pv = state_info[8]
        expected_grid = next_load - next_pv  # 예상 그리드 상태 계산
        exp_grid_act = expected_grid + action / 4

        # 충전 및 방전 가능 범위 계산
        max_charge = float(min(env.conv_cap, (env.soc_max - current_soc) * env.battery_cap))
        max_discharge = float(min(env.conv_cap, (current_soc - env.soc_min) * env.battery_cap))

        # exp_grid_act가 Tensor인지 확인하고 필요시 Tensor로 변환
        if not isinstance(exp_grid_act, torch.Tensor):
            exp_grid_act = torch.tensor(exp_grid_act, dtype=torch.float32)

        # 역송 방지 및 SoC 제약 조건 통합
        if exp_grid_act < 0:  # 역송이 예상되는 경우
            if current_soc < env.soc_max:  # 충전할 여유가 있는 경우
                action = torch.min(
                    torch.tensor(max_charge, dtype=torch.float32, device=action.device),
                    (-4 * exp_grid_act).clone().detach()
                )
            else:  # SoC가 최대 용량에 도달하여 더 이상 충전할 수 없는 경우
                action = torch.tensor(0.0, dtype=torch.float32).to(action.device)  # 대기 상태로 전환
        else:  # 일반적인 SoC 제약 조건 및 방전 시 역송 방지 조건 적용
            if action < 0:  # 방전 시 역송 방지
                max_discharge_to_prevent_reverse = min(-4 * (next_load - next_pv), max_discharge)
                action = max(action, max_discharge_to_prevent_reverse)

            # 일반 SoC 제약 조건 적용
            action = torch.clamp(action, -max_discharge, max_charge)

        # Action scaling 적용
        action = action * scale

        # Action에 대한 로그 확률 계산
        action_log_prob = dist.log_prob(action).sum(dim=-1)

        # Action 반환 (Tensor -> Numpy 변환)
        return action.detach().cpu().numpy(), action_log_prob

    def act(self, state):
        dist, _ = self.forward(state)
        action = dist.sample()
        action = torch.clamp(action, -max_action, max_action)
        action = torch.round(action).int()  # 소수점을 제거하고 정수로 변환
        action_log_prob = dist.log_prob(action.float()).sum(dim=-1)  # 정수형을 float으로 변환 후 log_prob 계산
        return action.detach().cpu().numpy(), action_log_prob

class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim=hidden_dim, lr=learning_rate, gamma=gamma, eps_clip=eps_clip, K_epochs=K_epochs, max_action=max_action):
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim, max_action)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, state, memory, env, state_info):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 상태를 GPU로 이동 tensor
        action, action_log_prob = self.policy.act(state)    # ndarray

        # 현재 SoC 값
        current_soc = state_info[0] # ndarray
        next_load = state_info[7]  # next_load float
        next_pv = state_info[8]  # next_pv float
        expected_grid = next_load - next_pv
        exp_grid_act = expected_grid + action / 4

        # 충전/방전 범위를 계산
        max_charge = min(env.conv_cap, (env.soc_max - current_soc) * env.battery_cap)
        max_discharge = min(env.conv_cap, (current_soc - env.soc_min) * env.battery_cap)
        torch_discharge = torch.tensor(max_discharge, dtype=torch.float32).detach().to(device)
        torch_charge = torch.tensor(max_charge, dtype=torch.float32).detach().to(device)
        # print(torch_charge, type(torch_charge))

        # 역송 방지 제약 조건 추가
        # SoC 제약 조건 및 역송 방지 조건 통합
        if expected_grid < 0:  # 역송이 예상되는 경우
            if current_soc < env.soc_max:  # 충전할 여유가 있는 경우
                action = min(max_charge, (-4 * expected_grid))  # 역송된 전력을 충전
            else:  # SoC가 최대 용량에 도달하여 더 이상 충전할 수 없는 경우
                action = 0  # 대기 상태
        elif exp_grid_act < 0:  # 수동 역송이 예상되는 경우
            if current_soc <= env.soc_max:  # 충전 여유가 있는 경우
                action = min(max_charge, (-4 * exp_grid_act))  # 수동 역송 방지를 위해 충전
            elif current_soc >= env.soc_min:  # 방전 가능할 경우
                # 역송이 발생하지 않는 방전량 계산
                max_discharge_to_prevent_reverse = -4 * exp_grid_act
                max_discharge_to_prevent_reverse = max(max_discharge_to_prevent_reverse, -max_discharge)
                action = max(action, max_discharge_to_prevent_reverse)  # 역송 방지 범위로 방전 제한
        else:  # 일반적인 SoC 제약 조건 및 방전 시 역송 방지 조건 적용
            if action < 0:  # 방전 시 역송 방지
                max_discharge_to_prevent_reverse = min(-4 * (next_load - next_pv), max_discharge)
                action = max(action, max_discharge_to_prevent_reverse)  # 역송 방지 범위로 방전 제한

            # 일반 SoC 제약 조건 적용
            if action > max_charge:
                action = torch_charge
            elif action < -max_discharge:
                action = -torch_discharge

        # 출력 디버깅 (필요 시 사용)
        # print(f'Step {env.current_step}: SoC: {current_soc}, Action: {action}, Expected Grid: {expected_grid}')

        # # Tensor 자료형을 ndarray로 변환
        if isinstance(action, torch.Tensor):
            action = action.clone().cpu().numpy().reshape(1, 1)
        if isinstance(action, float):
            action = np.array([[action]], dtype=float)

        # Action에 대한 로그 확률 계산 및 메모리에 저장
        memory.append_state(state)
        memory.append_action(action)
        memory.append_log_prob(action_log_prob)

        return action  # 스칼라 값으로 반환

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize the rewards
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert memory to tensors and move to GPU
        old_states = torch.squeeze(torch.stack(memory.states)).detach().to(device)
        old_actions = torch.tensor(np.array(memory.actions), dtype=torch.float32).detach().to(device)
        old_log_probs = torch.tensor(memory.log_probs).detach().to(device)

        batch_size = batch_num  # 미니배치 크기 설정
        for _ in range(self.K_epochs):
            for i in range(0, len(memory.states), batch_size):
                # 미니배치 생성
                batch_states = old_states[i:i + batch_size]
                batch_actions = old_actions[i:i + batch_size]
                batch_log_probs = old_log_probs[i:i + batch_size]
                batch_rewards = rewards[i:i + batch_size]

                # Evaluating old actions and values
                log_probs, state_values = self.evaluate(batch_states, batch_actions)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(log_probs - batch_log_probs.detach())

                # Finding Surrogate Loss
                advantages = batch_rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * (state_values - batch_rewards) ** 2

                # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

    def evaluate(self, state, action):
        dist, state_value = self.policy(state)
        action_log_probs = dist.log_prob(action).sum(dim=-1)
        return action_log_probs, torch.squeeze(state_value)

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def append_state(self, state):
        if isinstance(state, torch.Tensor):
            self.states.append(state.clone().detach())
        else:
            self.states.append(torch.tensor(state, dtype=torch.float32))
    def append_action(self, action):
        # Ensure action is a tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        self.actions.append(action)

    def append_log_prob(self, log_prob):
        # Ensure log_prob is a tensor
        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor(log_prob, dtype=torch.float32)
        self.log_probs.append(log_prob)

    def append_reward(self, reward):
        # Ensure reward is a tensor
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32)
        self.rewards.append(reward)

    def append_is_terminal(self, is_terminal):
        # Ensure is_terminal is a tensor (bool type)
        if not isinstance(is_terminal, torch.Tensor):
            is_terminal = torch.tensor(is_terminal, dtype=torch.bool)
        self.is_terminals.append(is_terminal)


def plot_results(dates, load, pv, grid, action, soc, peak_times, title):
    # 데이터를 스칼라 값으로 변환 (평탄화)
    grid = [float(value) if isinstance(value, (list, np.ndarray)) else value for value in grid]
    load = [float(value) if isinstance(value, (list, np.ndarray)) else value for value in load]
    pv = [float(value) if isinstance(value, (list, np.ndarray)) else value for value in pv]
    action = [float(value) if isinstance(value, (list, np.ndarray)) else value for value in action]
    soc = [float(value) if isinstance(value, (list, np.ndarray)) else value for value in soc]

    # 모든 데이터가 동일한 길이인지 확인
    assert len(dates) == len(load) == len(pv) == len(grid) == len(action) == len(soc), "데이터 길이가 일치하지 않습니다."

    # 데이터가 스칼라 값으로 이루어져 있는지 검사
    for i, value in enumerate(grid):
        assert np.isscalar(value), f"grid 데이터의 {i}번째 요소가 스칼라 값이 아닙니다. 값: {value}, 타입: {type(value)}"

    plt.figure(figsize=(14, 8))

    # 첫 번째 그래프: Load, PV, Grid (왼쪽 y축), SoC (오른쪽 y축)
    ax1 = plt.subplot(2, 1, 1)

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

    plt.title(f'{title} Episodes')
    plt.tight_layout()
    plt.show()

def train_multi_ppo(env, charge_agent, discharge_agent, charge_memory, discharge_memory, episodes=episodes, save_interval=save_interval, max_timesteps=max_timesteps, batch_interval=batch_interval):

    episode_rewards = []
    actions = []
    best_total_cost = float('inf')  # 최소 total_cost를 추적
    best_model_path = "best_ppo.pth"  # 최적 모델 저장 경로

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
        timestep = 0

        dates = []
        load_values = []
        pv_values = []
        grid_values = []
        soc_values = []
        action_values = []
        peak_values = []

        for t in range(max_timesteps):

            # 에이전트가 환경으로부터 액션을 선택
            # 충전 에이전트 행동 선택
            charge_action = charge_agent.select_action(state, charge_memory, env, state_info)
            charge_action = max(0, charge_action)  # 충전 에이전트 행동 제한

            # 방전 에이전트 행동 선택
            discharge_action = discharge_agent.select_action(state, discharge_memory, env, state_info)
            discharge_action = min(0, discharge_action)  # 방전 에이전트 행동 제한

            # 환경에 두 행동 전달
            actions = [charge_action, discharge_action]
            next_state, reward, done, next_state_info = env.step(actions)

            # 메모리에 보상과 종료 여부를 저장
            charge_memory.append_reward(reward)
            discharge_memory.append_reward(reward)
            charge_memory.append_is_terminal(done)
            discharge_memory.append_is_terminal(done)

            total_reward += reward

            # 데이터를 저장
            dates.append(env.date)
            load_values.append(state_info[1])
            pv_values.append(state_info[2])
            grid_values.append(state_info[3])
            soc_values.append(state_info[0])
            action_values.append(actions)
            peak_values.append(env.peak_time)

            # 상태를 업데이트
            state = next_state
            state_info = next_state_info
            timestep += 1

            # batch_interval마다 업데이트 및 메모리 초기화
            if timestep % batch_interval == 0:
                # print('ACT: ', memory.actions)
                # print('STAT: ', memory.states)
                # print('len', len(memory.actions), len(memory.states))

                # 충전 에이전트 업데이트
                charge_agent.update(charge_memory)
                charge_memory.clear_memory()

                # 방전 에이전트 업데이트
                discharge_agent.update(discharge_memory)
                discharge_memory.clear_memory()

            # 에피소드 종료 체크
            if done:
                break

        # 에피소드가 끝날 때마다 정책 업데이트
        if timestep % batch_interval != 0:
            # 충전 에이전트 업데이트
            charge_agent.update(charge_memory)
            charge_memory.clear_memory()

            # 방전 에이전트 업데이트
            discharge_agent.update(discharge_memory)
            discharge_memory.clear_memory()

        # agent.update(memory)
        # memory.clear_memory()

        # 에피소드 보상 기록
        episode_rewards.append(total_reward)

        if env.total_cost < best_total_cost:
            best_total_cost = env.total_cost
            torch.save(agent.policy.state_dict(), best_model_path)
            print(f"New best model saved at Episode {episode + 1} with total_cost: {env.total_cost}")

            # 최적 모델의 데이터를 저장
            best_episode_data["dates"] = dates
            best_episode_data["load"] = load_values
            best_episode_data["pv"] = pv_values
            best_episode_data["grid"] = grid_values
            best_episode_data["action"] = action_values
            best_episode_data["soc"] = soc_values
            best_episode_data["peak_time"] = peak_values

        # 에피소드 정보 출력
        print(
            f'Episode {episode + 1}/{episodes} finished with reward: {total_reward.item() if isinstance(total_reward, np.ndarray) else float(total_reward):.2f} / {env.total_cost}')
        # 모델 저장
        if (episode + 1) % save_interval == 0:
            torch.save(agent.policy.state_dict(), f"multi_{episode + 1}.pth")

    # 학습 보상 시각화
    episode_rewards.pop(0)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward over Episodes(Multi)')
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
        'Best'
    )

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
    peak_values = []

    total_reward = 0

    # 테스트 루프
    for t in range(env.total_steps):
        # 추론 시에는 탐색을 비활성화합니다
        if test == 1:
            with torch.no_grad():
                action, _ = agent.policy.test_act(torch.FloatTensor(state).unsqueeze(0), env, state_info, scale)
        else:
            with torch.no_grad():
                action, _ = agent.policy.act(torch.FloatTensor(state).unsqueeze(0))

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

    # 테스트 결과 시각화
    plot_results(date_values, load_values, pv_values, grid_values, action_values, soc_values, peak_values, model_path)
    return cost_info

test_csv = pd.read_csv('gloom_work.csv')
work_env = Environment(workday_csv)
hol_env = Environment(holiday_csv)
test_env = Environment(d0907_csv)
env = hol_env

# 메모리 초기화
charge_memory = Memory()
discharge_memory = Memory()
# 충전 에이전트
charge_agent = PPO(state_dim, action_dim, hidden_dim, max_action=250)
# 방전 에이전트
discharge_agent = PPO(state_dim, action_dim, hidden_dim, max_action=250)

# GPU 설정 (초기화 이후 어디서든 가능합니다)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
charge_agent.policy.to(device)  # 모델을 GPU로 이동
discharge_agent.policy.to(device)

# 학습 및 추론
train_multi_ppo(env, charge_agent, discharge_agent, charge_memory, discharge_memory, max_timesteps=env.total_steps)
run_test(env, f"multi_{episodes}.pth", ppo_agent, 1, 0)
# run_test(env, f"custom_4000.pth", ppo_agent, 1, 0, 1)