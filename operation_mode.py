import config as cfg
import datetime
import pandas as pd
import random
import mqtt_fn
import db_fn

def optimize_mode(soc):
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')

    date_str = (cfg.now + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')

    predWL = float(round(load.loc[load['date'] == date_str, 'load'][0], 2))
    predWPV = float(pv.loc[cfg.now.hour]['pv'])

    # soc = cfg.soc_index[mqtt_fn.pms_index]
    #soc = db_fn.get_pms_soc()

    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL

    print('predL: {}, predPV: {}, soc: {}'.format(predWL, predWPV, soc))
    print('wcnd : {}, wsoc : {}({}%)'.format(round(wcnd,1), wsoc, soc))

    if wcnd < 0: # 방전, 음수
        if wsoc >= cfg.min_capacity:
            if cfg.min_capacity > wsoc + wcnd:  # soc 10% 미만 방지
                if wcnd <= -cfg.conv_capacity_1h:  # 컨버터 용량 초과 방지
                    return -(wsoc - cfg.min_capacity)
                else:
                    return -cfg.conv_capacity_1h
            else:                                   # PF 값 선정
                if wcnd <= -cfg.conv_capacity_1h:   # 컨버터 용량 초과 방지
                    return -cfg.conv_capacity_1h
                else:
                    return wcnd
        else:   # 배터리 soc 10% 미만일 경우
            return 0
    elif wcnd > 0:   # 충전 양수
        if cfg.max_capacity < wsoc + wcnd:   # soc 90% 초과 방지
            return cfg.max_capacity - wsoc
        else:
            if wcnd >= cfg.conv_capacity_1h:   # 컨버터 용량 초과 방지
                return cfg.conv_capacity_1h
            else:
                return wcnd
    else:  # 대기
        return 0

#print(f'optimize pref={optimize_mode(21.5)}')
for i in range(89,99):
    print(f'soc={i}, p_ref={optimize_mode(i)}')


def peak_mode(limit):
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')
    date_str = (cfg.now + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')
    predWL = float(load.loc[(load['date'] == date_str), 'load'])
    predWPV = float(pv.loc[cfg.now.hour]['pv'])

    soc = db_fn.get_pms_soc()
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL
    hour = datetime.datetime.now().hour
    peak_rate = 0.1

    if 22 <= hour or hour < 8:  # 충전 22시 ~ 08시
        if cfg.max_capacity > wsoc:    # soc 90% 초과 방지
            if cfg.max_capacity - wsoc >= cfg.conv_capacity_1h:   # 컨버터 용량 초과 방지
                return cfg.conv_capacity_1h
            else:
                return cfg.max_capacity - wsoc
        else:               # 충전 불가능
            return 0
    else:       # 방전 08시 ~ 21시
        if -wcnd > limit:   # 피크 초과시
            if wsoc >= cfg.min_capacity: # 배터리 잔여 전력 확인
                if cfg.min_capacity > wsoc + wcnd:   # soc 10% 미만 방지
                    if wcnd <= -cfg.conv_capacity_1h:  # 컨버터 용량 초과 방지
                        return -cfg.conv_capacity_1h
                    elif -wcnd + (wcnd + limit) * peak_rate < wsoc :
                        return (wcnd + limit) * peak_rate
                    else:
                        return -(wsoc - cfg.min_capacity)
                else:   # 피크치 만큼 방전
                    if (wcnd + limit) * peak_rate <= -cfg.conv_capacity_1h:  # 컨버터 용량 초과 방지
                        return -cfg.conv_capacity_1h
                    else:
                        return (wcnd + limit) * peak_rate # 피크치 보다 10% 더 방전
            else:   # ESS에 잔여 전력 없을시
                return 0
        else:   # 피크 초과 안할시
            return 0



def demand_mode():

    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')
    date_str = (cfg.now + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')
    predWL = float(load.loc[(load['date'] == date_str), 'load'])
    predWPV = float(pv.loc[cfg.now.hour]['pv'])

    soc = db_fn.get_pms_soc()
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL
    hour = datetime.datetime.now().hour

    print('predL: {}, predPV: {}'.format(predWL, predWPV))
    print('wcnd : {}, wsoc : {}({}%)'.format(round(wcnd,1), wsoc, soc))

    if 22 <= hour or hour < 8:  # 충전 22시 ~ 08시 (경부하)
        if cfg.max_capacity > wsoc:    # soc 90% 초과 방지
            if cfg.max_capacity - wsoc >= cfg.conv_capacity_1h:   # 컨버터 용량 초과 방지
                return cfg.conv_capacity_1h
            else:
                return cfg.max_capacity - wsoc
        else:               # 충전 불가능
            return 0
    elif 8 <= hour < 16:  # 대기 08시 ~ 16시 (중간부하)
        return 0
    else:           # 방전 16시 ~ 22시 (최대부하)
        if cfg.min_capacity >= wsoc + wcnd:  # soc 10% 이상 및 미만 방지 100 > 130-40(90)
            if -wcnd <= cfg.conv_capacity_1h:  # 컨버터 최대 용량 제한
                return -cfg.conv_capacity_1h
            else:  # soc 10% 까지만 방전
                return -(wsoc - cfg.min_capacity)
        else:
            if -wcnd >= cfg.conv_capacity_1h:  # 컨버터 최대 용량 제한
                return -cfg.conv_capacity_1h
            else:  # soc 10% 까지만 방전
                return wcnd


def pv_mode():

    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')
    date_str = (cfg.now + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')
    predWL = float(load.loc[(load['date'] == date_str), 'load'])
    predWPV = float(pv.loc[cfg.now.hour]['pv'])

    soc = db_fn.get_pms_soc()
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL
    hour = datetime.datetime.now().hour
    print('predL: {}, predPV: {}'.format(predWL, predWPV))
    print('wcnd : {}, wsoc : {}({}%)'.format(round(wcnd,1), wsoc, soc))

    if 10 <= hour < 16:  # 충전 10시 ~ 16시
        if predWPV > 0 and wsoc < cfg.max_capacity:
            if cfg.max_capacity >= wsoc + predWPV:  # soc 90% 이하 및 초과 방지   900 + 200
                return predWPV
            elif wsoc < cfg.max_capacity:   # soc 90% 까지만 충전
                return cfg.max_capacity - wsoc
            else:  # 충전 불가능 soc 90% 이상
                return 0
        else :
            return 0
    else:     # 그외 나머지 시간 방전
        if wcnd < 0 and wsoc > cfg.min_capacity:  # 방전, 음수
            if cfg.min_capacity >= wsoc + wcnd:    # soc 10% 이상 및 미만 방지 100 > 130-40(90)
                if -wcnd <= cfg.conv_capacity_1h:  # 컨버터 최대 용량 제한
                    return -cfg.conv_capacity_1h
                else:   # soc 10% 까지만 방전
                    return -(wsoc - cfg.min_capacity)
            else:
                if -wcnd >= cfg.conv_capacity_1h:  # 컨버터 최대 용량 제한
                    return -cfg.conv_capacity_1h
                else:   # soc 10% 까지만 방전
                    return wcnd
        else:   # 방전 불가능 soc 10% 미만
            return 0
