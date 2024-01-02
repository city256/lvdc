import datetime
from pytimekr import pytimekr
import pandas as pd
import numpy as np


def check_date(date_str):
    # 문자열에서 날짜 객체로 변환
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    # 해당 날짜가 공휴일인지 확인
    holidays = pytimekr.holidays(date_obj.year)

    # 주말인지 확인
    if date_obj.weekday() >= 5:  # 토요일
        return 1
    elif date_obj in holidays:
        return 1
    return 0

# 예제
#print(datetime.datetime.today().date())
#date_str = str(datetime.datetime.today().date())  # 원하는 날짜를 입력
#print(check_date(date_str))



def insert_price(csv_name):
    df = pd.read_csv(csv_name)
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = df.apply(calculate_price, axis=1)
    df.to_csv(csv_name, index=False)

def calculate_price(row):
    month = row['date'].month
    hour = row['date'].hour

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

insert_price('../pqms_data_dqn.csv')

def insert_fee(csv_name):
    contracted = 500 # 계약전력
    base_rate = 8320 # 고압A 선택2 기본요금
    base_fee = contracted * base_rate

    df = pd.read_csv(csv_name)
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = df.apply(calculate_price, axis=1)
    df.to_csv(csv_name, index=False)