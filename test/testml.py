import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import db_fn

# 데이터 로드, 여기서는 'df'라는 이름의 데이터프레임을 가정합니다.
df = db_fn.get_test()
df = df.set_index('date')

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['load'].values.reshape(-1,1))

# 시퀀스 데이터 생성 함수
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# 시퀀스 길이 설정 (가령, 과거 24시간 데이터를 기반으로 미래를 예측)
look_back = 24

# 훈련용 / 테스트용 데이터 분리
train_size = int(len(scaled_data) * 0.75)
test_size = len(scaled_data) - train_size
train, test = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

# 데이터셋 생성
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# LSTM 입력을 위한 데이터 shape 변환
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# LSTM 모델 생성
model = Sequential()
model.add(LSTM(200, input_shape=(1, look_back)))
model.add(Dense(1))

# 모델 컴파일
model.compile(loss='mean_squared_error', optimizer='adam')

# 모델 훈련
model.fit(trainX, trainY, epochs=30, batch_size=30, verbose=1)

# 테스트 데이터에 대한 예측값 생성
testPredict = model.predict(testX)

# 예측값 스케일 역변환
testPredict = scaler.inverse_transform(testPredict)

# 예측하려는 날짜 설정
predict_until = pd.to_datetime('2023-07-30 00')

# 예측값을 저장할 빈 리스트 생성
predictions = []

# 현재까지의 전체 데이터 사용
current_data = np.copy(scaled_data)

while len(predictions) < (predict_until - df.index[-1]).total_seconds() / 3600:
    # 가장 최근 데이터를 바탕으로 예측 수행
    sample = current_data[-look_back:]
    sample = np.reshape(sample, (1, 1, look_back))

    # 모델을 사용하여 예측 수행
    predicted_power_usage = model.predict(sample)

    # 예측값을 predictions 리스트에 추가
    predictions.append(predicted_power_usage[0][0])

    # 현재 데이터에 예측값 추가
    current_data = np.append(current_data, predicted_power_usage)

# 예측값 스케일 역변환
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 예측 시간 생성
prediction_dates = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), predict_until, freq='H')

# 예측 날짜와 전력 사용량을 데이터프레임으로 변환
prediction_df = pd.DataFrame(data={
    'date': prediction_dates,
    'load': predictions.flatten()  # 2D array를 1D array로 변환
})

print(prediction_df)
