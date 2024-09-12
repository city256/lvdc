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
        self.battery_cap = 1000  # Battery capacity in kWh
        self.conv_cap = 250  # Converter capacity in kW
        self.contract = 500
        self.soc_min = 0.1  # Minimum SoC (10%)
        self.soc_max = 0.9  # Maximum SoC (90%)
        self.total_steps = len(data)
        self.demand_cost = 8320  # Demand cost in your local currency
        self.cnd_state = [0]
        self.episode_reward = 0
        self.current_step = 0
        self.current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        self.demand_sum = 0

        # 충전 및 방전 효율
        self.charging_efficiency = 0.9  # 90% efficiency
        self.discharging_efficiency = 0.9  # 90% efficiency

        # 모델링 변수 설정
        self.done = False
        self.soc = 0.5  # 초기 SoC 설정 (50% charged)
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
        self.generation = 0

        self.usage_cost = 0  # max(0, self.grid) * self.price
        self.usage_sum = 0
        self.total_cost = 0  # (self.peak * self.demand_cost + self.usage_sum) * 1.137

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

        self.generation = 0
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

    def calculate_soc(self, previous_soc, action):
        """
        Calculate the new state of charge (SoC) based on the action taken and the battery capacity.
        Charging and discharging efficiency is applied here.
        """
        if action > 0:  # Charging
            # Apply charging efficiency
            energy_added = (action / 4) * self.charging_efficiency
            new_soc = previous_soc + energy_added / self.battery_cap
        elif action < 0:  # Discharging
            # Apply discharging efficiency
            energy_removed = (abs(action) / 4) / self.discharging_efficiency
            new_soc = previous_soc - energy_removed / self.battery_cap
        else:
            new_soc = previous_soc  # No change if no action

        # Ensure SoC is within bounds
        return max(self.soc_min, min(self.soc_max, new_soc))

    def step(self, action):
        """
        주어진 액션에 따라 환경의 상태를 업데이트하고,
        다음 상태, 보상, 에피소드 종료 여부를 반환합니다.
        """
        current_date = self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'date'].iloc[0]
        current_load = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'load'].iloc[0])
        current_pv = float(self.data.loc[self.data['Unnamed: 0_x'] == self.current_step, 'pv'].iloc[0])
        current_grid = current_load - current_pv + (action / 4)
        current_soc = self.calculate_soc(self.soc, action)  # Using the new calculate_soc function
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
        self.peak_time = current_peak_time
        self.total_cost = (self.demand_sum + self.usage_sum) * 1.137

        # 역송된 전력량
        if self.load - self.pv < 0:
            self.generation += self.load - self.pv

        # 마지막 스텝의 reward 계산 및 step + 1
        if self.current_step >= self.total_steps - 1:  # 마지막 스텝에서 종료
            self.done = True
            reward = (current_usage_cost)  # 사용 요금을 누적해서 보상에 반영
            print('Done!!!')
        else:
            if action > 0:  # 충전
                if current_peak_time == 0:  # 저렴한 시간대
                    reward = (action / 4) * current_price * 1.5  # 충전 시 보상 증가
                elif current_peak_time == 2:  # 비싼 시간대
                    reward = - (current_usage_cost)
                else:
                    reward = (action / 4) * current_price
            elif action < 0:  # 방전
                if current_peak_time == 2:  # 비싼 시간대
                    reward = - (current_usage_cost * 0.5)  # 방전 시 손해 증가
                elif current_peak_time == 0:  # 저렴한 시간대
                    reward = - (current_usage_cost * 1.2)
                else:
                    reward = - (current_usage_cost)
            else:
                reward = - (current_usage_cost)

            # 상태 업데이트
            self.current_step += 1

        self.episode_reward += reward
        next_state = [self.soc, self.load, self.pv, self.grid, -self.total_cost, self.switch_sum, self.peak,
                      self.peak_time, self.workday]

        self.render(action)
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

    def is_working_hour(self, current_date):
        """
        주어진 날짜/시간이 근무 시간(T_work)에 속하는지 여부를 반환합니다.
        """
        dt = datetime.datetime.strptime(current_date, '%Y-%m-%d %H:%M:%S')
        hour = dt.hour
        return 9 <= hour <= 18  # 예시로 9시~18시를 근무 시간으로 가정

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



import matplotlib.pyplot as plt
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


def apply_constraints(current_state, action, env):
    soc, load, pv, grid, total_cost, switch_sum, peak, peak_time, workday = current_state

    # 현재 스텝의 다음 스텝 데이터를 가져옴
    next_step = env.current_step  # 다음 스텝 인덱스를 증가시킴
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


def apply_normal(current_state, action, env):
    soc, load, pv, grid, total_cost, switch_sum, peak, peak_time, workday = current_state

    # 현재 스텝의 다음 스텝 데이터를 가져옴
    next_step = env.current_step  # 다음 스텝 인덱스를 증가시킴
    if next_step < len(env.data):
        next_load = float(env.data.loc[env.data['Unnamed: 0_x'] == next_step, 'load'].iloc[0])
        next_pv = float(env.data.loc[env.data['Unnamed: 0_x'] == next_step, 'pv'].iloc[0])
    else:
        next_load = load
        next_pv = pv

    # 충전/방전 후 예상 SoC 계산
    expected_soc = soc + (action / 4) / env.battery_cap  # action은 kW 단위, SoC는 비율
    expected_grid = next_load - next_pv
    expected_grid_action = expected_grid + (action / 4)

    # 충전 조건: PV > Load (역송 발생) & SoC가 최대값 미만
    if expected_grid < 0 and soc < env.soc_max:
        print('reverse!')
        charge_amount = min(-expected_grid * 4, (env.soc_max - soc) * env.battery_cap * 4)  # 역송된 만큼 충전
        action = max(charge_amount, 0) * 1.1
        print(f'Charging: {action} kW, SoC: {soc} -> {expected_soc}')
    # 방전 조건: 근무 시간대(T_work) & SoC가 최소값 초과
    elif env.is_working_hour(env.current_date) and soc > env.soc_min and env.workday:
        action = max(-20, -(soc - env.soc_min) * env.battery_cap * 4)  # 방전량 고정 (20kW)
        print(f'Discharging: {action} kW, SoC: {soc} -> {expected_soc}')
    else:
        # 대기: 근무 시간 외 or SoC가 최소/최대 제약 조건을 넘지 않는 경우
        action = 0
        print(f'Waiting: No Action, SoC: {soc} -> {expected_soc}')



    # Grid 계산이 제대로 이루어지는지 확인하기 위해 출력
    print(f"[{env.current_step}] Next Step: {next_step}, Load: {next_load}, PV: {next_pv}, Grid: {expected_grid}")

    return action


def apply_peak(current_state, action, env):
    soc, load, pv, grid, total_cost, switch_sum, peak, peak_time, workday = current_state

    # 현재 스텝의 다음 스텝 데이터를 가져옴
    next_step = env.current_step  # 현재 스텝 인덱스를 그대로 사용
    if next_step < len(env.data):
        next_load = float(env.data.loc[env.data['Unnamed: 0_x'] == next_step, 'load'].iloc[0])
        next_pv = float(env.data.loc[env.data['Unnamed: 0_x'] == next_step, 'pv'].iloc[0])
    else:
        next_load = load
        next_pv = pv

    # 예상 그리드 전력 계산 (부하 - PV)
    pred_grid = next_load - next_pv
    peak_thresh = 30  # 임계값(피크 제한값) 설정

    # 충전 조건: 역송 발생 시 (PV > Load) & SoC가 최대값 미만 & 경부하 시간대
    if (pred_grid < 0 and soc < env.soc_max):
        if pred_grid < 0:
            charge_amount = min(abs(pred_grid) * 4, (env.soc_max - soc) * env.battery_cap * 4)  # 역송된 전력량만큼 충전
        else:
            charge_amount = min((peak_thresh - pred_grid), (env.soc_max - soc) * env.battery_cap * 4)  # 충전량 계산
        action = max(charge_amount, 0)  # 충전량은 양수여야 하므로 max 적용
        print(f'Charging: {action} kW, SoC: {soc} -> {soc + (action / 4) / env.battery_cap}')
    elif env.peak_time == 0 and soc < env.soc_max:
        print('경부하')
        charge_amount = min((peak_thresh - pred_grid), (env.soc_max - soc) * env.battery_cap * 4)  # 충전량 계산
        action = max(charge_amount, 0)  # 충전량은 양수여야 하므로 max 적용
    # 방전 조건: 예상 그리드 전력이 피크 임계값을 초과하고 SoC가 최소값을 초과하는 경우
    elif pred_grid > peak_thresh and soc > env.soc_min:
        # 방전량은 피크 임계값까지 줄이도록 설정
        discharge_amount = min((pred_grid - peak_thresh) * 4, (soc - env.soc_min) * env.battery_cap * 4)
        action = -discharge_amount  # 방전량 제한 (최대 20kW)
        print(f'Discharging: {action} kW, SoC: {soc} -> {soc + (action / 4) / env.battery_cap}')

    else:
        # 충전 및 방전 조건을 모두 만족하지 않는 경우 대기
        action = 0
        print(f'Waiting: No Action, SoC: {soc}')

    # 상태 업데이트 및 그리드 변화 계산
    expected_grid_action = next_load - next_pv + (action / 4)  # 그리드에 충방전 액션 반영
    print(
        f"[{env.current_step}] Action: {action}, Load: {next_load}, PV: {next_pv}, Grid: {expected_grid_action}, SoC: {soc}")

    return action
def apply_demand(current_state, action, env):
    soc, load, pv, grid, total_cost, switch_sum, peak, peak_time, workday = current_state

    # 현재 스텝의 다음 스텝 데이터를 가져옴
    next_step = env.current_step  # 다음 스텝 인덱스를 그대로 사용
    if next_step < len(env.data):
        next_load = float(env.data.loc[env.data['Unnamed: 0_x'] == next_step, 'load'].iloc[0])
        next_pv = float(env.data.loc[env.data['Unnamed: 0_x'] == next_step, 'pv'].iloc[0])
    else:
        next_load = load
        next_pv = pv

    # 예상 그리드 전력 계산 (부하 - PV)
    pred_grid = next_load - next_pv

    # 1. 역송 방지 우선
    # If predicted PV is greater than load, we charge the battery to prevent reverse flow
    if next_pv > next_load and soc < env.soc_max:
        charge_amount = min((next_pv - next_load) * 4, (env.soc_max - soc) * env.battery_cap * 4)
        action = max(charge_amount, 0)
        print(f"Charging to prevent reverse flow: {action} kW, SoC: {soc} -> {soc + (action / 4) / env.battery_cap}")

    # 2. 충전/방전/대기 조건을 설정
    elif env.peak_time == 0:  # 경부하 시간대 (Off-peak period)
        if soc < env.soc_max:
            # 충전하는 경우 (SOC가 최대 충전 용량에 도달하지 않은 경우)
            charge_amount = min(100, (env.soc_max - soc) * env.battery_cap * 4)
            action = max(charge_amount, 0)
            print(f'Charging: {action} kW, SoC: {soc} -> {soc + (action / 4) / env.battery_cap}')
        else:
            action = 0  # 대기 상태
            print(f'Waiting: No Action, SoC is full')

    elif env.peak_time == 1:  # 중간부하 시간대 (Mid-peak period)
        # 중간부하 시간대에는 대기
        action = 0
        print(f'Mid-Peak Period: No Action, SoC: {soc}')

    elif env.peak_time == 2:  # 최대부하 시간대 (On-peak period)
        if soc > env.soc_min and pred_grid > 0:  # 최대부하 시간대 & 부하 발생 시 방전
            discharge_amount = min((pred_grid) * 4, (soc - env.soc_min) * env.battery_cap * 4)
            action = max(-discharge_amount, -env.conv_cap)  # ESS 변환기 용량으로 방전량 제한
            print(f'Discharging: {action} kW, SoC: {soc} -> {soc + (action / 4) / env.battery_cap}')
        else:
            action = 0  # 대기 상태
            print(f'Waiting: No Action, SoC is low or no load')

    # 상태 업데이트 및 그리드 변화 계산
    expected_grid_action = next_load - next_pv + (action / 4)
    print(f"[{env.current_step}] Action: {action}, Load: {next_load}, PV: {next_pv}, Grid: {expected_grid_action}, SoC: {soc}")

    return action



data_csv = pd.read_csv('../test/0907_0911.csv')

# 환경 초기화
env = Environment(data_csv)
state = env.reset()

pv_values = []
load_values = []
grid_values = []
soc_values = []
action_values = []
date_values = []

for t in range(env.total_steps):
    current_date = env.current_date
    date_values.append(current_date)
    load_values.append(env.load)
    pv_values.append(env.pv)
    grid_values.append(env.grid)
    soc_values.append(env.soc)

    action = apply_constraints(state, 0, env)
    action_values.append(action)

    next_state, reward, done = env.step(action)
    print(f'[{t}] {next_state}')
    if env.done:
        break
    state = next_state

plot_results(date_values, load_values, pv_values, grid_values, action_values, soc_values, 0)

print(env.total_steps)