import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, LSTM
import matplotlib.pyplot as plt
import time
import numpy as np

print('entering load')
start_time = time.time()

# CSV 파일 로드, db_fn.get_load_data_15() 대신 pd.read_csv 사용
df = pd.read_csv('load_data.csv')
df['date'] = pd.to_datetime(df['date'])

# 데이터 전처리
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute

# NaN이 아닌 부분을 훈련 데이터로 사용
train_df = df.dropna()

# Feature와 label 분리
features = train_df[['hour', 'minute', 'workday']].values
labels = train_df['load'].values

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features)
labels = labels / max(labels)

# GRU에 입력할 데이터 형태로 변환
X, y = [], []
n_past = 672
for i in range(n_past, len(features)):
    X.append(features[i - n_past:i])
    y.append(labels[i])

X = np.array(X)
y = np.array(y)

# 데이터 분할
split_index = int(len(X) * 0.8)  # 80%는 훈련, 20%는 테스트
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# GRU 모델 구성
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_past, 3)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')

# 모델 훈련
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# 예측
predicted = model.predict(X_test)
predicted = predicted.reshape(-1)  # 차원 변경
predicted = predicted * max(train_df['load'])  # 역정규화


# 실제 값과 예측 값을 그래프로 비교
plt.figure(figsize=(15, 5))
plt.plot(y_test * max(train_df['load']), label='Actual Load', color='blue')
plt.plot(predicted, label='Predicted Load', color='red')
plt.title('GRU Load Prediction')
plt.xlabel('Time')
plt.ylabel('Load')
plt.legend()
plt.show()

print('load done : ', time.time() - start_time)