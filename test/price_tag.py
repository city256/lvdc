import datetime
import numpy as np
import pandas as pd

def calculate_price(datetime_str):
    dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
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


x=2
if x ==1 :
    name = 'normal'
    df = pd.read_csv('../csv/pqms_data_normal.csv')
elif x==2:
    name = 'peak'
    df = pd.read_csv('../csv/pqms_data_peak.csv')
elif x==3:
    name = 'demand'
    df = pd.read_csv('../csv/pqms_data_demand.csv')
elif x==4:
    name = 'pv'
    df = pd.read_csv('../csv/pqms_data_pv.csv')

df['date'] = pd.to_datetime(df['date'])  # Replace 'datetime' with your date column name
df['fee'] = df['date'].apply(lambda x: calculate_price(x.strftime('%Y-%m-%d %H:%M:%S')))
df['price'] = df['acdc'] * df['fee']
money = df['price'].sum()
grid = df['acdc'].sum()
discharge = df['ess_discharge'].sum()
charge = df['ess_charge'].sum()

print(name)
print(f'fee = {round(money,2)}, grid = {grid}, charge = {round(charge,2)}/{round(discharge,2)}')
#df.to_csv('pqms_data_pv.csv', index=False)  # This will save the DataFrame with the new 'fee' column

