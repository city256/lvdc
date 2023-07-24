import db
import mqtt_fn
import config as cfg
import datetime
import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.by import By
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import re

# 기상청 태양광발전량 예측
url = 'http://bd.kma.go.kr/kma2020/fs/energySelect2.do?menuCd=F050702000'
pv_capasity = 0.25 # (mW) 0.25 = 250kW

now = datetime.datetime.now()
now_date = datetime.datetime.now().strftime('%Y-%m-%d')
now_hour = datetime.datetime.now().strftime('%Y-%m-%d %H')


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def predict_load():
    start_time = time.time()
    # 데이터 로드, 여기서는 'df'라는 이름의 데이터프레임을 가정합니다.
    df = pd.read_csv('load6_2020.csv')

    # '날짜'를 datetime으로 변환하고 인덱스로 설정
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.set_index('date')

    # 데이터 정규화
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['load'].values.reshape(-1, 1))

    # 시퀀스 길이 설정 (가령, 과거 24시간 데이터를 기반으로 미래를 예측)
    look_back = 24

    # 훈련용 / 테스트용 데이터 분리
    train_size = int(len(scaled_data) * 0.75)
    test_size = len(scaled_data) - train_size
    train, test = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

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

    # 모델 훈련 (GPU 사용)
    with tf.device("/device:GPU:0"):
        model.fit(trainX, trainY, epochs=50, batch_size=12, verbose=1)

    # 테스트 데이터에 대한 예측값 생성
    testPredict = model.predict(testX)

    # 예측값 스케일 역변환
    testPredict = scaler.inverse_transform(testPredict)
    df_actual = pd.read_csv('load1_2021.csv')

    # '날짜'를 datetime으로 변환하고 인덱스로 설정
    df_actual['date'] = pd.to_datetime(df_actual['date'], format='%Y-%m-%d')
    df_actual = df_actual.set_index('date')

    # 현재 시각

    # 예측 하려는 시간 설정

    # 실제 데이터 로드
    df_actual = pd.read_csv('load1_2021.csv')

    # '날짜'를 datetime으로 변환하고 인덱스로 설정
    df_actual['date'] = pd.to_datetime(df_actual['date'])
    df_actual = df_actual.set_index('date')

    # 예측하려는 날짜 설정
    predict_until = pd.to_datetime('2021-01-07 23')

    # 예측값을 저장할 빈 리스트 생성
    predictions = []

    # 예측 시간 생성
    prediction_dates = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), predict_until, freq='H')

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

    # 예측값 pd 파일 생성
    pred_load = pd.DataFrame()

    for i in range(len(predictions)):
        temp = pd.DataFrame(data=[[prediction_dates[i], round(predictions[i][0]*10,2)]], columns=['date', 'load'])
        pred_load = pd.concat([pred_load, temp], ignore_index=True)

    '''
        error = []
        # 날짜 설정
        for i in range(24):
            for j in range(1, 8):
                date_to_show = pd.to_datetime('2021-01-' + str(j).zfill(2) + ' ' + str(i).zfill(2))

                # 해당 날짜의 실제 전력 사용량 출력
                actual_power_usage = df_actual.loc[date_to_show, 'load']
                #print(f'Actual power usage at {date_to_show}: {actual_power_usage}')

                # 해당 날짜의 예측된 전력 사용량 출력
                predicted_power_usage = predictions[prediction_dates.get_loc(date_to_show)]
                #print(f'Predicted power usage at {date_to_show}: {round(predicted_power_usage[0], 2)}')

                # 오차율 계산
                error_rate = (abs(actual_power_usage - predicted_power_usage[0]) / actual_power_usage) * 100
                error.append(error_rate)

                # 오차율 출력
                #print(f'Error rate at {date_to_show}: {error_rate:.2f}%')

        print(f'average error rate : {np.mean(error):.2f}%')
        print(f'Training Time : {time.time() - start_time}')
    '''
    pred_load.to_csv('pred_load.csv')
    return pred_load

def predict_pv():

    # 크롬창 백그라운드 실행
    option = webdriver.ChromeOptions()
    option.add_argument('headless')
    option.add_argument('disable-gpu')
    driver = webdriver.Chrome(options=option)

    # 웹페이지 접속
    driver.get(url)

    # 웹페이지 로딩 대기
    driver.implicitly_wait(3)

    # 지역 선택 및 클릭
    driver.find_element(By.ID, 'install_cap').clear()
    driver.find_element(By.ID, 'install_cap').send_keys(pv_capasity)
    driver.find_element(By.ID, 'txtLat').send_keys(34.982)
    driver.find_element(By.ID, 'txtLon').send_keys(126.690)
    driver.find_element(By.ID, 'search_btn').click()
    time.sleep(10)

    # 웹페이지의 HTML 가져오기
    html = driver.page_source

    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(html, 'html.parser')

    pred_pv = pd.DataFrame(columns=['date', 'pv'])
    sunlight = driver.find_element(By.ID, 'toEnergy').text
    hour = sunlight.split('\n')

    i = 0
    for x in hour:
        energy = x.split(' ')
        strtime = datetime.datetime.strptime(now_date+" "+re.sub('[^A-Za-z0-9]', '', energy[0]), '%Y-%m-%d %H')
        # 특정 시간대 값이 누락 되었을 때
        if energy[1] == '-':
            pred_pv.loc[i] = [strtime, energy[1]]
        else:
            pred_pv.loc[i] = [strtime, float(energy[1]) * 1000]
        i += 1
    pred_pv.to_csv('pred_pv.csv')
    return pred_pv

def data_preprocessing():
    pv = predict_pv()
    load = predict_load()

    return

def calculate_pf(limit=None, time=None, pf=None):
    pv = predict_pv()
    load =  predict_load()

    if (limit!=None):
       print(limit)
    elif(time!=None):
        print(time)
    elif(pf!=None):
        print(pf)
    else:
        print(pv)
        print(load)

    pf = 12.1

    return pf

def optimize_mode():
    #pv = predict_pv()
    #load = predict_load()
    #pv['date'] = pd.to_datetime(pv['date'], format='%Y-%m-%d %H')
    #load['date'] = pd.to_datetime(load['date'], format='%Y-%m-%d %H')
    #predWL = load['date'].loc[now_hour+':00:00']
    #predWPV = float(pv.loc[pv['date'] == now_hour, 'pv'])
    predWPV = 130
    predWL = 394.8
    soc = 14.6
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWL - predWPV
    print(wcnd, wsoc)

    if wcnd > 0: # 방전
        if wsoc >= cfg.min_capacity:
            if cfg.min_capacity < wsoc + wcnd:
                return wsoc - cfg.min_capacity
            else:
                if wcnd >= cfg.conv_capacity_1h:
                    return cfg.conv_capacity_1h
                else:
                    return wcnd
        else:
            return 0
    elif wcnd < 0:   # 충전
        if cfg.max_capacity < wsoc + wcnd:
            return cfg.max_capacity - wsoc
        else:
            if wcnd <= -cfg.conv_capacity_1h:
                return -cfg.conv_capacity_1h
            else:
                return wcnd
    else:  # 대기
        return 0

def peak_mode(limit):
    predWPV = 130
    predWL = 394.8
    soc = 30.6
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWL - predWPV

    if 8 < int(datetime.datetime.hour) and int(datetime.datetime.hour) > 22:
        if cfg.max_capacity < wsoc + wcnd:
            return cfg.max_capacity - wsoc
        else:
            return cfg.conv_capacity_1h
    else:
        if wcnd > limit:
            if wsoc >= cfg.min_capacity:
                if cfg.min_capacity > wsoc + wcnd:
                    return wsoc - cfg.min_capacity
                else:
                    return wcnd - limit
            else:
                return 0
        else:
            return 0

def demand_mode():
    predWPV = predict_pv()
    predWL = predict_load()

    wsoc = 30  # 현재 soc양 pms에 요청
    wcnd = predWL - predWPV


    print("demand_mode")
    return 13.3

def pv_mode():
    pv = predict_pv()
    load = predict_load()
    print("pv_mode")
    return 13.4
