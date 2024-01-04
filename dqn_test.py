import datetime
import random
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from datetime import timedelta
import db_fn
from tensorflow.keras.models import load_model
import pandas as pd

total_episodes = 100
batch_size = 32
start_time = (datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))).strftime('%Y-%m-%d %H:%M')


def calculate_price(datetime_str):
    dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
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

class EMSEnvironment:
    def __init__(self, data):
        self.data = data
        self.current_index = 0
        self.current_time = (datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))).strftime('%Y-%m-%d %H:%M')
        self.max_index = 0
        self.price = calculate_price(self.current_time)
        self.soc = 50
        self.load = float(round(self.data.iloc[self.current_index]['load'], 2))
        self.pv = float(round(self.data.iloc[self.current_index]['pv'], 2))
        # 다른 필요한 변수들을 추가할 수 있습니다.

    def reset(self):
        self.current_index = 0
        self.max_index = len(self.data)
        self.current_time = self.current_time = self.data.iloc[self.current_index]['date']
        self.price = calculate_price(self.current_time)
        self.sum_discharge=0
        self.sum_charge = 0
        self.sum_grid = 0
        self.sum_fee = 0
        self.soc = db_fn.get_pms_soc()
        self.load = float(round(self.data.iloc[self.current_index]['load'], 2))
        self.pv = float(round(self.data.iloc[self.current_index]['pv'], 2))
        return np.array([self.soc, self.load, self.pv, self.price])

    def step(self, action):

        self.load = float(round(self.data.iloc[self.current_index]['load'], 2))
        self.pv = float(round(self.data.iloc[self.current_index]['pv'], 2))
        print(self.current_index, self.current_time, self.load, self.pv)
        # 여기에 실제 시스템에 대한 행동의 영향을 계산하는 로직을 구현합니다.
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
        self.current_index += 1
        self.soc += (action/4) * 0.1  # 예시: 행동에 따라 SOC 업데이트
        self.load = float(round(self.data.iloc[self.current_index]['load'], 2))
        self.pv = float(round(self.data.iloc[self.current_index]['pv'], 2))

        self.sum_discharge += discharge
        self.sum_charge += charge
        self.sum_grid += grid
        self.sum_fee += reward
        self.current_time = self.data.iloc[self.current_index]['date']
        next_state = np.array([self.soc, self.load, self.pv, self.price])

        # 에피소드 종료 조건을 체크합니다 (예: 24시간을 넘어갔을때)
        if self.current_index >= self.max_index - 1:
            done = True
        print(f'grid= {grid}, load= {self.load}, pv= {self.pv}, charge= {charge}/{discharge}, soc= {self.soc}%')
        return next_state, reward, done


model_save_path = 'dqn_model.h5'
#agent.save(model_save_path)

# 모델 불러오기
model = load_model(model_save_path)

# 새로운 데이터 파일 로드
new_data_file = 'pqms_data_15_dqn.csv'
new_data = pd.read_csv(new_data_file)

# 상태 크기 정의 (SOC, Load, PV, Price)
state_size = 4

# 결과를 저장할 리스트 초기화
results = []

# 초기 SOC 값 설정
soc = 50  # 예시 값, 실제 초기 SOC에 따라 조정 필요

for index, row in new_data.iterrows():
    # 현재 상태 생성
    load = row['load']  # 'load' 컬럼
    pv = row['pv']  # 'pv' 컬럼
    price = calculate_price(row['date'])  # 'datetime' 컬럼과 calculate_price 함수 사용
    state = np.array([[soc, load, pv, price]])

    # 모델을 사용하여 행동 예측
    act_values = model.predict(state)
    action = np.argmax(act_values[0]) - 250  # 행동 조정

    # SOC 업데이트 (실제 시스템에 따라 조정 필요)
    soc += (action / 4) * 0.1  # 예시: 행동에 따라 SOC 업데이트
    soc = max(0, min(100, soc))  # SOC를 0과 100 사이로 제한

    # 결과 저장
    results.append({'index': index, 'date': row['date'], 'soc': soc, 'action': action})

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)
print(results_df)