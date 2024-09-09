import pandas as pd
import datetime
from pytimekr import pytimekr

train_csv = pd.read_csv('../test/train_1day.csv')
test_csv = pd.read_csv('../test/testset.csv')

# hyperparameter
w1 = 0.3 # cost weight
w2 = 0.3 # peak weight
w3 = 0.4 # swtich weight

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
        self.current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        self.demand_sum = 0

        # 모델링 변수 설정
        self.done = False
        self.soc = 0 # 초기 SoC 설정 (충전 상태)
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

        self.usage_cost = 0 # max(0, self.grid) * self.price
        self.usage_sum = 0
        self.total_cost = 0 #(self.peak * self.demand_cost + self.usage_sum) * 1.137

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
        self.grid = self.load - self.pv
        self.price, self.peak_time = calculate_price(self.current_date)
        self.workday = check_workday(self.current_date)
        self.demand_sum = 0

        self.charge = 0
        self.discharge = 0
        self.ess_action = 0
        self.peak = self.grid
        self.switch_sum = 0
        self.usage_cost = max(0, self.grid) * self.price
        self.usage_sum = self.usage_cost
        self.total_cost = (self.peak * self.demand_cost + self.usage_sum) * 1.137
        self.cnd_state = [0]
        self.episode_reward = 0

        state = [self.soc, self.load, self.pv, self.grid, -self.total_cost, self.switch_sum, self.peak, self.peak_time, self.workday]
        return state

    def step(self, action):
        """
        주어진 액션에 따라 환경의 상태를 업데이트하고,
        다음 상태, 보상, 에피소드 종료 여부를 반환합니다.
        """
        current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        current_load = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'load'].iloc[0])
        current_pv = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'pv'].iloc[0])
        current_grid = current_load - current_pv + (action / 4)
        current_soc = calculate_soc(self.soc, action, self.battery_cap)
        current_price, current_peak_time = calculate_price(self.current_date)
        current_usage_cost = current_price * max(self.grid, 0)
        current_peak = calculate_peak(self.peak, current_load, current_pv, action, self.contract)

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

        self.charge = current_charge
        self.discharge = current_discharge
        self.ess_action = action
        self.current_date = current_date
        self.load = current_load
        self.pv = current_pv
        self.grid = current_grid
        self.price = current_price
        self.soc = current_soc
        self.workday = check_workday(self.current_date)
        self.peak = current_peak
        self.usage_cost = current_usage_cost
        self.usage_sum += current_usage_cost
        self.switch_sum += calculate_switch(action, self.cnd_state)
        self.demand_sum = self.peak * self.demand_cost
        self.total_cost = (self.demand_sum + self.usage_sum) * 1.137

        # 보상 계산


        # 마지막 스텝의 reward 계산 및 step + 1
        if self.current_step >= self.total_steps - 1:  # 마지막 스텝에서 종료
            self.done = True
            reward = -self.total_cost  # 에피소드가 끝나면 total_cost에 따라 패널티/보상
            print('Done!!!')
        else:
            # 상태 업데이트
            if action > 0:  # 충전
                if current_peak_time == 0:  # 저렴한 시간대
                    reward = current_usage_cost   # 충전 시 보상 증가
                elif current_peak_time == 2:  # 비싼 시간대
                    reward = - (current_usage_cost * 1.5)
                else:
                    reward = - (current_usage_cost)
            elif action < 0:  # 방전
                if current_peak_time == 2:  # 비싼 시간대
                    reward = - (current_usage_cost * 0.5)  # 방전 시 손해 증가
                elif current_peak_time == 0:  # 저렴한 시간대
                    reward = - (current_usage_cost * 1.5)
                else:
                    reward = - (current_usage_cost)
            else:
                reward = - (current_usage_cost)
            # grid가 음수일 때 이를 양수로 만드는 액션에 추가 보상
            if self.load - self.pv < 0 and current_grid >= 0:
                reward += 10 * (current_grid * current_price)  # grid가 양수로 변경되면 보상 추가
                print('prevent reverse')

            self.current_step += 1

        self.episode_reward += reward
        next_state = [self.soc, self.load, self.pv, self.grid, -self.total_cost, self.switch_sum, self.peak,
                      self.peak_time, self.workday]

        return next_state, reward, self.done

    def render(self, action):
        # 현재 상태를 출력
        print(f"Step: {self.current_step}")
        print(f"[{self.current_date}] SoC: {round(self.soc*100,2)}%")
        print(f"Load: {self.load}, PV: {self.pv}, Grid : {round(self.grid,2)}, Price: {self.price}")
        print(f"Total: {round(self.total_cost,0)}, Demand: {round(self.demand_cost*self.peak,0)}, Usage: {round(self.usage_sum, 0)}")
        print(f"Switch Sum: {self.switch_sum}, Peak: {self.peak}")
        print(f'action: {action}')
        print("-" * 40)

def cost_normalize(data, min_val, max_val):
    # 최대값과 최소값이 동일할 경우 0 반환
    if max_val == min_val:
        return 0
    return (data - min_val) / (max_val - min_val)
def switch_normalize(data, min_val, max_val):
    # 최대값과 최소값이 동일할 경우 0 반환
    if max_val == min_val:
        return 0
    return (data - min_val) / (max_val - min_val)

def peak_normalize(data, min_val, max_val):
    # 최대값과 최소값이 동일할 경우 0 반환
    if max_val == min_val:
        return 0
    return (data - min_val) / (max_val - min_val)
def z_score_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0  # 표준편차가 0인 경우 모든 값이 동일하므로 0으로 반환
    return (data - mean) / std
# Min-max 정규화 함수
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return 0  # 최대값과 최소값이 같은 경우 0으로 반환
    return (data - min_val) / (max_val - min_val)

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

# def compute_reward(total_cost, peak, switch_sum, a=w1, b=w2, c=w3):
#     total_cost = cost_normalize(total_cost, 0 , 10000000)
#     peak = peak_normalize(peak, 0 , 500)
#     switch_sum = switch_normalize(switch_sum, 0, 1151 * 2 - 1)
#
#     reward = - (a * total_cost + b * peak + c * switch_sum)
#     return reward


def compute_cost_switch(cost, switch_sum, w):
    cost = cost_normalize(cost, 0 , 1000000)
    switch_sum = switch_normalize(switch_sum, 0, 957)

    reward = - (w * cost + (1-w) * switch_sum)
    return reward
def compute_cost(total_cost, peak, a=0.5, b=0.5):
    reward = - (a * total_cost + b * peak)
    return reward

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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

# MPS 디바이스 설정 (Apple Silicon GPU 사용)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Action space: -1 to 1
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.action_std = nn.Parameter(torch.ones(1, action_dim) * 0.7)  # 초기 탐험 강화 (higher exploration at start)


    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value

    def select_action(self, state, memory, env):
        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
        action_mean, _ = self.forward(state_tensor)
        action_std = self.action_std.expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action = dist.sample()  # 액션 샘플링
        action = torch.clamp(action, -1, 1)  # -1~1 사이로 클램핑
        action = action.item() * 250  # Scale action to range [-250, 250]
        print(f'before action: {action}')

        # Apply constraints to the selected action
        action = self.apply_constraints(state, action, env)
        print(f'constraints action: {action}')

        # 메모리가 존재하는 경우에만 상태와 액션을 저장
        if memory is not None:
            action_logprob = dist.log_prob(torch.tensor(action / 250).to(device)).sum(dim=-1)  # Normalized action
            memory.states.append(state_tensor)
            memory.actions.append(torch.tensor(action).to(device))
            memory.logprobs.append(action_logprob)

        return action

    # def apply_constraints(self, current_state, action, env):
    #     soc, load, pv, grid, total_cost, switch_sum, peak, peak_time, workday = current_state
    #
    #     # 현재 스텝의 다음 스텝 데이터를 가져옴
    #     next_step = env.current_step
    #     if next_step < len(env.data):
    #         next_load = float(env.data.loc[env.data['Unnamed: 0_x'] == next_step, 'load'].iloc[0])
    #         next_pv = float(env.data.loc[env.data['Unnamed: 0_x'] == next_step, 'pv'].iloc[0])
    #     else:
    #         # 다음 스텝이 범위를 벗어나면 현재 스텝의 데이터를 사용하거나, 적절한 값을 설정
    #         next_load = load
    #         next_pv = pv
    #
    #     # SoC 제약 조건 적용
    #     max_charge = min(env.conv_cap, ((env.soc_max - soc) * env.battery_cap)*4)
    #     max_discharge = min(env.conv_cap, ((soc - env.soc_min) * env.battery_cap)*4)
    #
    #     # 충전/방전 후 예상 SoC 계산
    #     expected_soc = soc + (action / 4) / env.battery_cap  # action은 kW 단위, SoC는 비율
    #     expected_grid = next_load - next_pv
    #     expected_grid_action = expected_grid + (action / 4)
    #     print(f'First action: {action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid}/{expected_grid_action}')
    #
    #     # 일반적인 SoC 제약 조건에 따른 충전/방전 제어
    #     if expected_soc < env.soc_min:
    #         # SoC가 최소값 이하로 떨어지지 않도록 조정
    #         action = max(-max_discharge, min(action, max_charge))
    #         expected_soc = soc + (action / 4) / env.battery_cap
    #         expected_grid_action = expected_grid + (action / 4)
    #         print(
    #             f'Adjusted action to maintain soc_min: {action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid_action}')
    #     elif expected_soc > env.soc_max:
    #         # SoC가 최대값 이상으로 올라가지 않도록 조정
    #         action = max(-max_discharge, min(action, max_charge))
    #         expected_soc = soc + (action / 4) / env.battery_cap
    #         expected_grid_action = expected_grid + (action / 4)
    #         print(
    #             f'Adjusted action to maintain soc_max: {action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid_action}')
    #     else:
    #         # 충전과 방전을 제약 조건 내에서만 허용
    #         action = max(-max_discharge, min(action, max_charge))
    #         expected_soc = soc + (action / 4) / env.battery_cap
    #         expected_grid_action = expected_grid + (action / 4)
    #         print(f'meaningful action: {action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid_action}')
    #
    #     # 작은 액션 제거 (노이즈 방지)
    #     if -20 < action < 20:
    #         action = 0
    #         print("Action is too small, setting to 0.")
    #
    #     # 최종 예상 SoC 및 Grid 업데이트
    #     expected_soc = soc + (action / 4) / env.battery_cap
    #     expected_grid_action = expected_grid + (action / 4)
    #     print(
    #         f'Load/PV: {next_load}/{next_pv}, Next grid: {expected_grid}, Grid with act: {expected_grid_action}, Expected SoC: {expected_soc}')
    #
    #     return action


    def apply_constraints(self, current_state, action, env):
        soc, load, pv, grid, total_cost, switch_sum, peak, peak_time, workday = current_state

        # 현재 스텝의 다음 스텝 데이터를 가져옴
        next_step = env.current_step + 1  # 다음 스텝 인덱스를 증가시킴
        if next_step < len(env.data):
            next_load = float(env.data.loc[env.data['Unnamed: 0_x'] == next_step, 'load'].iloc[0])
            next_pv = float(env.data.loc[env.data['Unnamed: 0_x'] == next_step, 'pv'].iloc[0])
        else:
            next_load = load
            next_pv = pv

        # SoC 제약 조건 적용
        max_charge = min(env.conv_cap, (env.soc_max - soc) * env.battery_cap)
        max_discharge = min(env.conv_cap, (soc - env.soc_min) * env.battery_cap)

        # 충전/방전 후 예상 SoC 계산
        expected_soc = soc + (action / 4) / env.battery_cap  # action은 kW 단위, SoC는 비율
        expected_grid = next_load - next_pv
        expected_grid_action = expected_grid + (action / 4)
        print(f'First action: {action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid}/{expected_grid_action}')

        # 역송 발생 시 충전으로 전환, 이때 SoC를 고려하여 충전량 제한
        if expected_grid < 0:
            if soc <= env.soc_max:
                action = min(max_charge, (-4 * expected_grid) * 1.1)  # 역송된 만큼 충전
                expected_soc = soc + (action / 4) / env.battery_cap
                expected_grid_action = expected_grid + (action / 4)
                print(f'Natural reverse: {action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid_action}')
            else:  # 충전 용량 없으면 대기
                action = 0
                expected_soc = soc + (action / 4) / env.battery_cap
                expected_grid_action = expected_grid + (action / 4)
                print(
                    f'Natural reverse, cant charge: {action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid_action}')
        else:
            # 랜덤 액션으로 인한 역송 발생 ==> soc_min 까지 방전 or 대기
            if expected_grid_action < 0:
                if expected_soc <= env.soc_min:  # 예상 soc가 10% 이하일때 soc_min 까지만 방전
                    max_safe_discharge = (soc - env.soc_min) * env.battery_cap * 4
                    action = max(-max_safe_discharge * 0.9, -max_discharge, (-4 * expected_grid) * 0.9)
                    expected_soc = soc + (action / 4) / env.battery_cap
                    expected_grid_action = expected_grid + (action / 4)
                    print(f'reverse action: {action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid_action}')
                else:  # soc <= soc_max
                    action = max(-(4 * expected_grid) * 0.9, -max_discharge)
                    expected_soc = soc
                    expected_grid_action = expected_grid + (action / 4)
                    print(f'reverse cant action: {action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid_action}')
            else:
                if expected_soc < env.soc_min:
                    required_action = (env.soc_min - soc) * env.battery_cap * 4
                    action = max(required_action, action)
                    expected_soc = soc + (action / 4) / env.battery_cap
                    expected_grid_action = expected_grid + (action / 4)
                    print(
                        f'Adjusted action to maintain soc_min: {action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid_action}')
                elif expected_soc > env.soc_max:
                    required_action = (env.soc_max - soc) * env.battery_cap * 4
                    action = min(required_action, action)
                    expected_soc = soc + (action / 4) / env.battery_cap
                    expected_grid_action = expected_grid + (action / 4)
                    print(
                        f'Adjusted action to maintain soc_max: {action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid_action}')
                else:
                    action = max(-max_discharge, min(action, max_charge))
                    expected_soc = soc + (action / 4) / env.battery_cap
                    expected_grid_action = expected_grid + (action / 4)
                    print(f'meaningful action:{action}, soc:{soc}/{expected_soc}, next_grid:{expected_grid_action}')

        # 작은 액션 제거 (노이즈 방지)
        if -20 < action < 20:
            action = 0
            print("Action is too small, setting to 0.")

        expected_soc = soc + (action / 4) / env.battery_cap
        expected_grid_action = expected_grid + (action / 4)
        print(
            f'Load/PV: {next_load}/{next_pv}, Next grid: {expected_grid}, Grid with act: {expected_grid_action}, Expected SoC: {expected_soc}')

        return action


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.agent = PPOAgent(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.loss_fn = nn.MSELoss()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device).detach()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)  # Normalization
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values = self.agent.forward(old_states)
            state_values = state_values.squeeze()  # 크기 조정
            advantages = rewards - state_values

            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.loss_fn(state_values, rewards)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def select_action(self, state, memory, env):
        return self.agent.select_action(state, memory, env)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def moving_average(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode='valid')


def plot_results(dates, load, pv, grid, action, soc, e_num):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 첫 번째 y축에 Load, PV, Grid 값을 플롯
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Load / PV / Grid Values')
    ax1.plot(dates, load, label="Load", color="blue", linestyle='-', linewidth=2)
    ax1.plot(dates, pv, label="PV", color="orange", linestyle='-', linewidth=2)
    ax1.plot(dates, grid, label="Grid", color="green", linestyle='-', linewidth=2)
    ax1.tick_params(axis='y')

    # 오른쪽 y축을 사용하여 SoC 값 플롯
    ax2 = ax1.twinx()
    ax2.set_ylabel('SoC Value')
    ax2.plot(dates, soc, label="SoC", color="purple", linestyle='--', linewidth=2)
    ax2.tick_params(axis='y')

    # 범례 추가 (두 y축에 대해 각각 추가)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 타이틀 설정
    plt.title(f"Episode : {e_num + 1}  Load, PV, Grid, Action, and SoC Over Time")

    # 그래프 표시
    plt.show()


def train_ppo(env, agent, memory, episodes=1000, save_interval=10, patience=50):
    episode_rewards = []  # 에피소드별 누적 리워드를 저장할 리스트
    best_reward = float('-inf')
    best_episode_data = None
    best_episode = -1
    best_cost_info = None
    best_model_state = None
    best_model_save_path = "best_model.pth"
    no_improvement_count = 0  # 개선되지 않은 에피소드 수를 카운트

    for episode in range(episodes):
        state = env.reset()

        # Step별 pv, load, grid, soc, action, dates를 저장할 리스트
        pv_values = []
        load_values = []
        grid_values = []
        soc_values = []
        action_values = []
        date_values = []
        total_reward = 0  # 에피소드 동안의 총 리워드

        # 에피소드 내에서 스텝을 진행
        for t in range(env.total_steps):
            action = agent.select_action(state, memory, env)
            next_state, reward, done = env.step(action)

            total_reward += reward  # 매 스텝의 보상을 누적
            memory.rewards.append(reward)  # 누적 보상
            memory.is_terminals.append(done)

            # Step별 데이터 저장
            pv_values.append(state[2])
            load_values.append(state[1])
            grid_values.append(state[3])
            soc_values.append(state[0])
            action_values.append(action)
            date_values.append(env.current_date)

            if done:
                break

            state = next_state

        # 마지막 보상에 total_cost 기반 추가 보상
        memory.rewards[-1] += -env.total_cost * 0.05

        # 에피소드가 끝난 후 업데이트
        agent.update(memory)
        memory.clear_memory()

        episode_rewards.append(total_reward)  # 에피소드별 총 보상을 저장

        # 최고 리워드 에피소드 갱신
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode_data = {
                'dates': date_values,
                'load': load_values,
                'pv': pv_values,
                'grid': grid_values,
                'soc': soc_values,
                'action': action_values,
            }
            best_episode = episode + 1
            best_cost_info = {
                'total_cost': env.total_cost,
                'usage_sum': env.usage_sum,
                'demand': env.demand_cost * env.peak,
                'switch_num': env.switch_sum,
                'reward': total_reward
            }
            best_model_state = agent.agent.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # 에피소드가 끝날 때 총 리워드를 출력
        print(f'Episode {episode + 1}/{episodes} finished with total reward: {total_reward:.2f}')

        # 10 에피소드마다 모델 저장 및 시각화
        if (episode + 1) % save_interval == 0:
            torch.save(agent.agent.state_dict(), f'test_ppo_{episode+1}.pth')
            plot_results(date_values, load_values, pv_values, grid_values, action_values, soc_values, episode)

        # 조기 종료 조건
        if no_improvement_count >= patience:
            print(f"Stopping early at episode {episode + 1}. No improvement in the last {patience} episodes.")
            break

    # 에피소드별 리워드 변화 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

    # 가장 높은 리워드를 기록한 에피소드 출력
    if best_episode_data is not None:
        print(f"\nBest Episode: {best_episode} with reward: {best_reward:.2f}")
        plot_results(
            best_episode_data['dates'],
            best_episode_data['load'],
            best_episode_data['pv'],
            best_episode_data['grid'],
            best_episode_data['action'],
            best_episode_data['soc'],
            0
        )

    # 최고 에피소드의 비용 정보 출력
    if best_cost_info is not None:
        print(f"\nCost information from best episode {best_episode}:")
        print(f"Total cost: {best_cost_info['total_cost']}")
        print(f"Usage sum: {best_cost_info['usage_sum']}")
        print(f"Demand cost: {best_cost_info['demand']}")
        print(f"Switch count: {best_cost_info['switch_num']}")
        print(f"Final reward: {best_cost_info['reward']}")

    # 최고 에피소드의 모델 상태 저장
    if best_model_state is not None:
        torch.save(best_model_state, best_model_save_path)
        print(f"\nBest model saved to {best_model_save_path}")


# 3. 저장된 모델로 다른 CSV 데이터를 기반으로 추론
def run_test(env, model_path, agent):
    # 모델 로드
    agent.agent.load_state_dict(torch.load(model_path))
    agent.agent.to(device)
    agent.agent.eval()

    state = env.reset()

    # Step별 pv, load, grid, soc, action, dates를 저장할 리스트
    pv_values = []
    load_values = []
    grid_values = []
    soc_values = []
    action_values = []
    date_values = []

    # 비용 정보 초기화
    total_reward = 0

    for t in range(env.total_steps):
        action = agent.select_action(state, None, env)  # agent를 통해 액션 선택
        next_state, reward, done = env.step(action)

        # Step별 데이터 저장
        pv_values.append(state[2])  # pv는 state의 세 번째 요소
        load_values.append(state[1])  # load는 state의 두 번째 요소
        grid_values.append(state[3])  # grid는 state의 네 번째 요소
        soc_values.append(state[0])  # soc는 state의 첫 번째 요소
        action_values.append(action)  # action 저장
        date_values.append(env.current_date)  # 날짜 저장

        total_reward = reward  # 마지막 스텝의 reward가 최종 total_cost에 기반한 보상임

        if done:
            break

        state = next_state

    # 비용 정보 출력
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

    # 결과 출력
    plot_results(date_values, load_values, pv_values, grid_values, action_values, soc_values, 0)

data_csv = pd.read_csv('../test/0907_0911.csv')

# 환경 초기화
env = Environment(data_csv)

# 하이퍼파라미터 설정
state_dim = 9  # state의 차원: [soc, load, pv, grid, total_cost, switch_sum, peak]
action_dim = 1  # 단일 action: 충/방전
hidden_dim = 128

# 에이전트 초기화
ppo_agent = PPO(state_dim, action_dim, hidden_dim, lr=0.001, gamma=0.98)
memory = Memory()

# 학습 후 최종 모델 저장
train_ppo(env, ppo_agent, memory, episodes=1000, save_interval=100, patience=300)

# 새로운 CSV 파일로 환경 생성 및 추론
test_csv = pd.read_csv('0907_0911.csv')
test_env = Environment(test_csv)
# 학습된 에이전트로 테스트 실행
#run_test(test_env, "test_ppo_1000.pth", ppo_agent)