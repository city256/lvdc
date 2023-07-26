import config as cfg
import datetime
import pandas as pd


def optimize_mode():
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')

    date_str = (cfg.now + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')

    predWL = float(load.loc[(load['date'] == date_str), 'load'])
    predWPV = float(pv.loc[cfg.now.hour]['pv'])

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
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')

    date_str = (cfg.now + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')

    predWL = float(load.loc[(load['date'] == date_str), 'load'])
    predWPV = float(pv.loc[cfg.now.hour]['pv'])

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
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')

    date_str = (cfg.now + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')

    predWL = float(load.loc[(load['date'] == date_str), 'load'])
    predWPV = float(pv.loc[cfg.now.hour]['pv'])

    print("demand_mode", predWL, predWPV)
    return 13.3

def pv_mode():
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')

    date_str = (cfg.now + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')

    predWL = float(load.loc[(load['date'] == date_str), 'load'])
    predWPV = float(pv.loc[cfg.now.hour]['pv'])

    print("pv_mode", predWL, predWPV)
    return 13.4