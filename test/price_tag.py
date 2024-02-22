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

peak = 0
for x in range(0,5):

    if x ==1 :
        name = 'normal'
        df = pd.read_csv('pqms_data_normal.csv')
    elif x==2:
        name = 'peak'
        df = pd.read_csv('pqms_data_peak.csv')
    elif x==3:
        name = 'demand'
        df = pd.read_csv('pqms_data_demand.csv')
    elif x==4:
        name = 'pv'
        df = pd.read_csv('pqms_data_pv.csv')
    else:
        name = 'origin'
        df = pd.read_csv('pqms_data_origin.csv')

    df['date'] = pd.to_datetime(df['date'])  # Replace 'datetime' with your date column name
    df['fee'] = df['date'].apply(lambda x: calculate_price(x.strftime('%Y-%m-%d %H:%M:%S')))
    df['price'] = df['acdc'] * df['fee']

    money = df['price'].sum()
    grid = df['acdc'].sum()
    discharge = df['ess_discharge'].sum()
    charge = df['ess_charge'].sum()
    pv = df['pv'].sum()
    load = df['load'].sum()


    print(name)
    # 전체 요금
    print(f'day fee = {round(money,2)}, grid = {round(grid,2)}, charge = {round(charge,2)}/{round(discharge,2)}, load = {round(load, 2)}, pv = {round(pv,2)}')

    start_date = pd.to_datetime('2023-11-03 10:00')
    end_date = pd.to_datetime('2023-11-03 18:00')

    work_time = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    money = work_time['price'].sum()
    grid = work_time['acdc'].sum()
    discharge = work_time['ess_discharge'].sum()
    charge = work_time['ess_charge'].sum()
    pv = work_time['pv'].sum()
    load = work_time['load'].sum()

    # 근무시간 요금
    print(f'worktime fee = {round(money,2)}, grid = {round(grid,2)}, charge = {round(charge,2)}/{round(discharge,2)}, load = {round(load, 2)}, pv = {round(pv,2)}')

    # 229 / 147.2 / 204.6
    on_peak = df[df['fee'] == 204.6]
    on_peak_money = on_peak['price'].sum()
    on_peak_grid = on_peak['acdc'].sum()
    on_peak_charge = on_peak['ess_charge'].sum()
    on_peak_discharge = on_peak['ess_discharge'].sum()
    on_peak_pv = on_peak['pv'].sum()
    on_peak_load = on_peak['load'].sum()

    # 146.9 / 116.5 / 147.1
    mid_peak = df[df['fee'] == 147.1]
    mid_peak_money = mid_peak['price'].sum()
    mid_peak_grid = mid_peak['acdc'].sum()
    mid_peak_charge = mid_peak['ess_charge'].sum()
    mid_peak_discharge = mid_peak['ess_discharge'].sum()
    mid_peak_pv = mid_peak['pv'].sum()
    mid_peak_load = mid_peak['load'].sum()

    # 94 / 94.0 / 101.0
    off_peak = df[df['fee'] == 101.0]
    off_peak_money = off_peak['price'].sum()
    off_peak_grid = off_peak['acdc'].sum()
    off_peak_charge = off_peak['ess_charge'].sum()
    off_peak_discharge = off_peak['ess_discharge'].sum()
    off_peak_pv = off_peak['pv'].sum()
    off_peak_load = off_peak['load'].sum()



    print(f'on_peak fee = {round(on_peak_money, 2)}, grid = {round(on_peak_grid,2)}, charge = {round(on_peak_charge, 2)}/{round(on_peak_discharge, 2)}, load = {round(on_peak_load, 2)}, pv = {round(on_peak_pv,2)}')
    print(f'mid_peak fee = {round(mid_peak_money, 2)}, grid = {round(mid_peak_grid,2)}, charge = {round(mid_peak_charge, 2)}/{round(mid_peak_discharge, 2)}, load = {round(mid_peak_load, 2)}, pv = {round(mid_peak_pv,2)}')
    print(f'off_peak fee = {round(off_peak_money, 2)}, grid = {round(off_peak_grid,2)}, charge = {round(off_peak_charge, 2)}/{round(off_peak_discharge, 2)}, load = {round(off_peak_load, 2)}, pv = {round(off_peak_pv,2)}')


    if x ==1 :
        df.to_csv('pqms_data_normal.csv', index=False)
    elif x==2:
        df.to_csv('pqms_data_peak.csv', index=False)
    elif x==3:
        df.to_csv('pqms_data_demand.csv', index=False)
    elif x==4:
        df.to_csv('pqms_data_pv.csv', index=False)
    else:
        df.to_csv('pqms_data_origin.csv', index=False)
