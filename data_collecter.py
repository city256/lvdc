import threading
import db_fn
import datetime
import time
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

def predict_load():
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

    # 테스트 데이터에 대한 예측값 생성
    testPredict = model.predict(testX)

    # 예측값 스케일 역변환
    scaler.inverse_transform(testPredict)

    # 예측하려는 날짜 설정
    predict_until = pd.to_datetime(now_hour) + datetime.timedelta(days=predict_range)

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

def predict_pv():
    start = time.time()
    print('entering pv_predict')

    # 크롬창 백그라운드 실행
    option = webdriver.ChromeOptions()
    option.add_argument('headless')
    option.add_argument('disable-gpu')
    driver = webdriver.Chrome(options=option)

    # 웹페이지 접속
    driver.get(url)

    # 웹페이지 로딩 대기
    driver.implicitly_wait(1)

    # 전라남도 지도 선택
    driver.find_element(By.XPATH, '//*[@id="map"]/div[5]/div[4]/div[13]').click()
    driver.implicitly_wait(1)

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
            pred_pv.loc[i] = [today_hour, float(weather[3]), float(weather[4])]
        if weather[8] == '-':
            pred_pv.loc[i+24] = [today_hour + datetime.timedelta(days=1), 0, 0]
        else:
            pred_pv.loc[i+24] = [today_hour + datetime.timedelta(days=1), float(weather[8]), float(weather[9])]
        i += 1

    # pred_pv.csv에 저장
    pred_pv = pred_pv.sort_index()
    pred_pv.to_csv('pred_pv.csv')
    print('pv done : ', time.time() - start)
    pass


import datetime
from pytimekr import pytimekr


def check_date(date_str):
    # 문자열에서 날짜 객체로 변환
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    # 해당 날짜가 공휴일인지 확인
    holidays = pytimekr.holidays(date_obj.year)

    # 주말인지 확인
    if date_obj.weekday() >= 5:  # 토요일
        return "주말"
    elif date_obj in holidays:
        return "공휴일"
    return "평일"


# 예제
print(datetime.datetime.today().date())
date_str = str(datetime.datetime.today().date())  # 원하는 날짜를 입력
print(check_date(date_str))


def update_csv():
    load_proc = threading.Thread(target=predict_load)
    pv_proc = threading.Thread(target=predict_pv)


    load_proc.start()
    pv_proc.start()

    load_proc.join()
    pv_proc.join()
    pass

update_csv()