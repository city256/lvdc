import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 모델 구조 정의 (학습 때 사용한 것과 동일해야 함)
def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    return model

# 상태 크기와 행동 크기 설정
state_size = 4  # 예시 값, 실제 상태 벡터의 크기에 맞게 조정
action_size = 501  # 예시 값, 실제 행동의 개수에 맞게 조정

# 모델 구조를 재구성하고 가중치 로드
model = build_model(state_size, action_size)
model.load_weights('../dqn_model.h5')

# 새로운 CSV 파일 읽기
new_data = pd.read_csv('../pqms_data_15_dqn.csv')

# 결과를 저장할 리스트 초기화
predictions = []
soc = 50

# 각 행에 대해 모델 예측 수행
for index, row in new_data.iterrows():
    # 현재 상태 벡터 생성 (예시, 실제 상태 벡터 구성에 따라 조정 필요)
    state = [soc, row['load'], row['pv'], row['price']]  # 필요한 특성 추가
    print(index, state)
    state = np.reshape(state, [1, -1])


    # 모델을 사용하여 예측
    action = model.predict(state)[0]

    # 예측 결과 저장
    predictions.append(action)

# 예측 결과를 데이터프레임에 추가
new_data['predictions'] = predictions

# 결과를 새로운 CSV 파일로 저장
new_data.to_csv('output_with_predictions.csv', index=False)
