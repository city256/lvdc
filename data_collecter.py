import threading
import db_fn
import datetime
from datetime import timedelta
import time
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
import config as cfg
import re

# 기상청 태양광발전량 예측
url = 'https://bd.kma.go.kr/kma2020/fs/energySelect1.do?pageNum=5&menuCd=F050701000'
pv_capasity = 0.25 # (mW) 0.25 = 250kW
now_date = datetime.datetime.now().strftime('%Y-%m-%d')
now_hour = datetime.datetime.now().strftime('%Y-%m-%d %H')
predict_range = 1 # days

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def old_predict_load():
    print('entering load')

    start_time = time.time()
    # 데이터 로드, 여기서는 'df'라는 이름의 데이터프레임을 가정합니다.
    df = db_fn.get_load_data()
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:00:00')
    df = df.set_index('date')

    # 데이터 정규화
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['load'].values.reshape(-1, 1))

    # 시퀀스 길이 설정 (가령, 과거 24시간 데이터를 기반으로 미래를 예측)
    look_back = 24

    # 훈련용 / 테스트용 데이터 분리
    train_size = int(len(scaled_data) * 0.75)
    train, test = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

    # 데이터셋 생성
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # LSTM 입력을 위한 데이터 shape 변환
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # LSTM 모델 생성
    model = Sequential()
    model.add(LSTM(100, input_shape=(1, look_back)))
    model.add(Dense(1))

    # 모델 컴파일
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 모델 훈련
    model.fit(trainX, trainY, epochs=50, batch_size=12, verbose=1)



    # 예측하려는 날짜 설정
    #predict_until = pd.to_datetime(now_hour) + datetime.timedelta(days=predict_range)
    predict_until = pd.to_datetime('2023-10-08 00:00:00')

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
    pred_load = pd.DataFrame(data={
        'date': prediction_dates,
        'load': predictions.flatten()  # 2D array를 1D array로 변환
    })
    pred_load.to_csv('pred_load.csv')
    print('load done : ',time.time() - start_time)
    pass

def predict_load_lstm():

    print('entering load')
    start_time = time.time()
    # LSTM
    # CSV 파일 로드
    df = db_fn.get_load_data()
    df['date'] = pd.to_datetime(df['date'])

    # 데이터 전처리
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute  # 15분 단위 데이터를 처리하기 위해 분도 추출

    # NaN이 아닌 부분을 훈련 데이터로 사용
    train_df = df.dropna()

    # Feature와 label 분리
    features = train_df[['hour', 'minute', 'workday']].values
    labels = train_df['load'].values

    # 데이터 정규화
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = scaler.fit_transform(features)
    labels = labels / max(labels)  # 간단하게 0~1 범위로 정규화

    # LSTM에 입력할 데이터 형태로 변환
    X, y = [], []
    n_past = 672  # 과거 x개 데이터를 사용
    for i in range(n_past, len(features)):
        X.append(features[i - n_past:i])
        y.append(labels[i])

    X = np.array(X)
    y = np.array(y)

    # LSTM 모델 구성
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_past, 3)))  # input_shape 수정
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # 모델 훈련
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)

    # NaN인 load 값을 예측
    for i in range(len(df) - 96, len(df)):
        input_data = df.loc[i - n_past:i - 1, ['hour', 'minute', 'workday']].values
        input_data = scaler.transform(input_data)  # 정규화
        predicted = model.predict(input_data.reshape(1, n_past, 3))  # reshape 및 예측
        df.loc[i, 'load'] = round(predicted[0][0] * max(train_df['load']), 2) # 정규화 해제 및 저장
    df.to_csv('pred_load.csv')
    print('load done : ', time.time() - start_time)
    pass


def predict_load_xgb():
    import xgboost as xgb

    print('entering load')
    start_time = time.time()

    # CSV 파일 로드
    df = db_fn.get_load_data_15()
    #df = pd.read_csv('pred_load_test.csv')
    df['date'] = pd.to_datetime(df['date'])

    # 데이터 전처리
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute  # 15분 단위 데이터를 처리하기 위해 분도 추출

    # NaN이 아닌 부분을 훈련 데이터로 사용
    train_df = df.dropna()

    # Feature와 label 분리
    features = train_df[['hour', 'minute', 'workday']].values
    labels = train_df['load'].values

    # 데이터 정규화
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = scaler.fit_transform(features)
    max_load = max(labels)
    labels = labels / max_load  # 간단하게 0~1 범위로 정규화

    # XGBoost 모델 훈련
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=5000, learning_rate=0.5, max_depth=30)
    model.fit(features, labels)

    # NaN인 load 값을 예측
    for i in range(len(df) - 96, len(df)):
        input_data = df.loc[i, ['hour', 'minute', 'workday']].values.reshape(1, -1)
        input_data = scaler.transform(input_data)  # 정규화
        if df.loc[i, 'workday'] == 0:
            df.loc[i, 'load'] = round(model.predict(input_data)[0] * max(train_df['load']), 2)  # workday가 0일 때 다른 값을 설정
        else:
            df.loc[i, 'load'] = round(model.predict(input_data)[0] * 75, 2)  # workday가 0이 아닐 때의 값

    df.to_csv('pred_load.csv')
    print('load done : ', time.time() - start_time)
    pass


def predict_load_rf():
    from sklearn.ensemble import RandomForestRegressor
    print('entering load')
    start_time = time.time()

    # CSV 파일 로드
    df = db_fn.get_load_data_15()
    df['date'] = pd.to_datetime(df['date'])

    # 데이터 전처리
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute  # 15분 단위 데이터를 처리하기 위해 분도 추출

    # NaN이 아닌 부분을 훈련 데이터로 사용
    train_df = df.dropna()

    # Feature와 label 분리
    features = train_df[['hour', 'minute', 'workday']].values
    labels = train_df['load'].values

    # 데이터 정규화 (옵션 - 필요에 따라 사용하시면 됩니다.)
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = scaler.fit_transform(features)
    labels = labels / max(labels)  # 간단하게 0~1 범위로 정규화

    # Random Forest 모델 훈련
    model = RandomForestRegressor(n_estimators=3000, random_state=2)
    model.fit(features, labels)

    # NaN인 load 값을 예측
    for i in range(len(df) - 96, len(df)):
        input_data = df.loc[i, ['hour', 'minute', 'workday']].values.reshape(1, -1)
        input_data = scaler.transform(input_data)  # 정규화 (옵션)
        df.loc[i, 'load'] = round(model.predict(input_data)[0] * max(train_df['load']), 2)  # 정규화 해제 및 저장

    df.to_csv('pred_load.csv')
    print('load done : ', time.time() - start_time)
    pass

def predict_pv(df):

    # 'hour' 특성 추가
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour

    # NaN이 아닌 부분을 훈련 데이터로 사용
    train_df = df.dropna()

    # Feature와 label 분리
    features = train_df[['sunlight', 'temperature', 'hour']].values
    labels = train_df['pv'].values

    # Random Forest 모델 훈련
    model = RandomForestRegressor(n_estimators=2000, random_state=22)
    model.fit(features, labels)

    # NaN인 pv 값을 예측
    nan_indexes = df[df['pv'].isna()].index
    for i in nan_indexes:
        input_data = df.loc[i, ['sunlight', 'temperature', 'hour']].values.reshape(1, -1)
        df.loc[i, 'pv'] = model.predict(input_data)[0]  # 예측 및 저장
    df.to_csv('pred_pv.csv')
    return df

def crawling_pv():
    start = time.time()
    print('entering pv_predict')

    # 크롬창 백그라운드 실행
    option = webdriver.ChromeOptions()
    option.add_argument('headless')
    option.add_argument('--headless')
    option.add_argument('--no-sandbox')
    option.add_argument('disable-gpu')
    driver = webdriver.Chrome(options=option)

    # 웹페이지 접속
    driver.get(url)

    # 웹페이지 로딩 대기
    driver.implicitly_wait(3)

    # 전라남도 지도 선택
    driver.find_element(By.XPATH, '//*[@id="map"]/div[5]/div[4]/div[13]').click()
    driver.implicitly_wait(5)

    # 지역 선택 및 클릭 (나주 태양광 발전량 선택 예제)
    '''driver.find_element(By.ID, 'install_cap').clear()
    driver.find_element(By.ID, 'install_cap').send_keys(pv_capasity)
    driver.find_element(By.ID, 'txtLat').send_keys(34.982)
    driver.find_element(By.ID, 'txtLon').send_keys(126.690)
    driver.find_element(By.ID, 'search_btn').click()
    time.sleep(7)'''

    # 기상청 예측 데이터  크롤링
    pv_energy = driver.find_element(By.ID, 'toEnergy').text
    split_pv_energy = pv_energy.split('\n')

    # 크롤링 데이터 pd.DataFrame 으로 변경 및 생성
    pred_pv = pd.DataFrame(columns=['date', 'sunlight', 'temperature'])
    i = 0
    for x in split_pv_energy:
        weather = x.split(' ')
        today_hour = datetime.datetime.strptime(now_date + " " + re.sub('[^A-Za-z0-9]', '', weather[0]), '%Y-%m-%d %H')

        # 특정 시간대 값이 누락 되었을 때
        if weather[3] == '-':
            pred_pv.loc[i] = [today_hour, 0, 0]
        else:
            pred_pv.loc[i] = [today_hour, float(weather[3]) * cfg.sunlight_scaling, float(weather[4])]
        if weather[8] == '-':
            pred_pv.loc[i+24] = [today_hour + datetime.timedelta(days=1), 0, 0]
        else:
            pred_pv.loc[i+24] = [today_hour + datetime.timedelta(days=1), float(weather[8]) * cfg.sunlight_scaling, float(weather[9])]
        i += 1

    db_pv = db_fn.get_pv_dataset()
    # pred_pv.csv에 저장
    pred_pv = pred_pv.sort_index()
    pred_pv['pv'] = np.nan
    db_pv = pd.concat([db_pv, pred_pv], ignore_index=True)
    # 테스트용 데이터 삽입
    #test_pd = pd.read_csv('pred_pv_test.csv')
    predict_pv(db_pv).to_csv('pred_pv.csv')
    print('pv done : ', time.time() - start)
    pass
#crawling_pv()
def update_csv():
    print(datetime.datetime.now(),'- start predict')
    start_time = time.time()
    load_proc = threading.Thread(target=predict_load_xgb())
    pv_proc = threading.Thread(target=crawling_pv)

    load_proc.start()
    pv_proc.start()

    load_proc.join()
    pv_proc.join()

    pass
