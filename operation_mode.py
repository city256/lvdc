import db
import mqtt_fn
import config as cfg
import datetime
import time
import data_collecter as dc

def optimize_mode():
    # pv = dc.predict_pv()
    # load = dc.predict_load()

    # 특정 시간대 값을 불러옴
    # pv['date'] = pd.to_datetime(pv['date'], format='%Y-%m-%d %H')
    # load['date'] = pd.to_datetime(load['date'], format='%Y-%m-%d %H')
    # predWL = load['date'].loc[now_hour+':00:00']
    # predWPV = float(pv.loc[pv['date'] == now_hour, 'pv'])
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
    predWPV = dc.predict_pv()
    predWL = dc.predict_load()

    wsoc = 30  # 현재 soc양 pms에 요청
    wcnd = predWL - predWPV


    print("demand_mode")
    return 13.3

def pv_mode():
    pv = dc.predict_pv()
    load = dc.predict_load()
    print("pv_mode")
    return 13.4
