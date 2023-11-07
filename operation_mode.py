import config as cfg
import datetime
import pandas as pd
import db_fn
from datetime import timedelta

#test_date = datetime.datetime.strptime('2023-10-23 17:45:00', '%Y-%m-%d %H:%M:00')


def optimize_mode():
    # 예측된 부하량, 발전량 변수 가져오기
    '''
    pv_date = (datetime.datetime.now()).strftime('%Y-%m-%d %H:00:00')
    load_date = (datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))).strftime('%Y-%m-%d %H:%M:00')
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')
    predWL = float(round(load.loc[load['date'] == load_date, 'load'].iloc[0], 2))
    predWPV = float(round(pv.loc[pv['date'] == pv_date, 'pv'].iloc[0] / 4, 2))
    hour = datetime.datetime.now().hour
    '''


    test_date = (datetime.datetime.now() - timedelta(minutes=datetime.datetime.now().minute % 15)).strftime('%Y-%m-%d %H:%M:00')
    pv = pd.read_csv('test_pv.csv', parse_dates=['date'])
    load = pd.read_csv('test_load.csv', parse_dates=['date'])
    predWL = float(round(load.loc[load['date'] == test_date, 'load'].iloc[0], 2))
    predWPV = float(round(pv.loc[pv['date'] == test_date, 'pv'].iloc[0], 2))
    hour = int(load.loc[load['date'] == test_date, 'hour'].iloc[0])
    workday = int(load.loc[load['date'] == test_date, 'workday'].iloc[0])
    print('predL: {}, predPV: {}'.format(predWL, predWPV))

    soc = db_fn.get_pms_soc()
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL

    if wcnd < 0: # 방전, 음수
        if 8 < hour < 18 and workday:  # 근무일, 근무시간 9~18시 확인
            if wsoc > cfg.min_capacity:  # wsoc가 soc_min + static Dis 이상인지 12% 이상인지
                return max(-cfg.static_discharge, cfg.min_capacity - wsoc)
            else:
                return 0
        else:
            return 0
    elif wcnd > 0:   # 충전 양수
        if wsoc < cfg.max_capacity:   # soc 90% 미만
            return min(wcnd, cfg.max_capacity - wsoc, cfg.conv_capacity_1h)
        else:    # 배터리 soc 90% 초과일 경우
            return 0   # print("SoC 90% 초과")
    else:  # 대기 cwnd = 0
        # print("wcnd = 0 대기")
        return 0

'''
    if wcnd < 0: # 방전, 음수
        if 8 < hour < 19 and db_fn.check_date(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:00')):  # 근무일, 근무시간 9~18시 확인
            if wsoc > cfg.min_capacity:  # wsoc가 soc_min + static Dis 이상인지 12% 이상인지
                return max(-cfg.static_discharge, cfg.min_capacity - wsoc)
            else:
                print('배터리 상한치')
                return 0
        else:
            print('근무시간외')
            return 0
    elif wcnd > 0:   # 충전 양수
        if wsoc < cfg.max_capacity:   # soc 90% 미만
            return min(wcnd, cfg.max_capacity - wsoc, cfg.conv_capacity_1h)
        else:    # 배터리 soc 90% 초과일 경우
            print('배터리 90% 이상')
            return 0   # print("SoC 90% 초과")
    else:  # 대기 cwnd = 0
        # print("wcnd = 0 대기")
        print('대기')
        return 0
    '''

def peak_mode(limit):
    # 예측된 부하량, 발전량 변수 가져오기
    test_date = (datetime.datetime.now() - timedelta(minutes=datetime.datetime.now().minute % 15)).strftime('%Y-%m-%d %H:%M:00')
    pv = pd.read_csv('test_pv.csv', parse_dates=['date'])
    load = pd.read_csv('test_load.csv', parse_dates=['date'])
    predWL = float(round(load.loc[load['date'] == test_date, 'load'].iloc[0], 2))
    predWPV = float(round(pv.loc[pv['date'] == test_date, 'pv'].iloc[0], 2))
    hour = int(load.loc[load['date'] == test_date, 'hour'].iloc[0])
    workday = int(load.loc[load['date'] == test_date, 'workday'].iloc[0])
    print('predL: {}, predPV: {}'.format(predWL, predWPV))

    soc = db_fn.get_pms_soc()
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL
    # peak_limit = limit # 함수 변수 사용
    peak_limit = cfg.peak_limit

    if 22 <= hour or hour < 8:  # 충전 22시 ~ 08시
        if cfg.max_capacity > wsoc:    # soc 90% 초과 방지
            return min(cfg.max_capacity - wsoc, 120)  # 최대 출력으로 충전
        else:  # 충전 불가능
            return 0
    else:       # 방전 8시 ~ 22시
        if -wcnd > peak_limit and 12 < hour < 18:   # 피크 초과시
            if wsoc >= cfg.min_capacity: # 배터리 잔여 전력 확인
                if cfg.min_capacity > wsoc + wcnd + peak_limit:   # soc 10% 미만 확인
                    return cfg.min_capacity - wsoc # 10% 까지만 방전
                else:   # 피크치 만큼 방전
                    return max(wcnd + peak_limit, -120) # 피크치까지 방전
            else:   # ESS에 잔여 전력 없을시
                return 0
        else:   # 피크 초과 안할시
            return 0

'''
    if workday:
        if 22 <= hour or hour < 9:  # 충전 22시 ~ 08시
            if cfg.max_capacity > wsoc:    # soc 90% 초과 방지
                return min(cfg.max_capacity - wsoc, 200)  # 최대 출력으로 충전
            else:  # 충전 불가능
                return 0
        else:       # 방전 09시 ~ 21시
            if -wcnd > peak_limit and 12 < hour < 18:   # 피크 초과시
                if wsoc >= cfg.min_capacity: # 배터리 잔여 전력 확인
                    if cfg.min_capacity > wsoc + wcnd + peak_limit:   # soc 10% 미만 확인
                        return cfg.min_capacity - wsoc # 10% 까지만 방전
                    else:   # 피크치 만큼 방전
                        return max(wcnd + peak_limit, -200) # 피크치까지 방전
                else:   # ESS에 잔여 전력 없을시
                    return 0
            else:   # 피크 초과 안할시
                return 0
    else:

    
'''

def demand_mode():

    '''
    # 예측된 부하량, 발전량 변수 가져오기
    pv_date = (datetime.datetime.now()).strftime('%Y-%m-%d %H:00:00')
    load_date = (datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))).strftime('%Y-%m-%d %H:%M:00')
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')
    predWL = float(round(load.loc[load['date'] == load_date, 'load'].iloc[0], 2))
    predWPV = float(round(pv.loc[pv['date'] == pv_date, 'pv'].iloc[0] / 4, 2))
    hour = datetime.datetime.now().hour
    '''

    # 예측된 부하량, 발전량 변수 가져오기
    test_date = (datetime.datetime.now() - timedelta(minutes=datetime.datetime.now().minute % 15)).strftime('%Y-%m-%d %H:%M:00')
    #test_date = datetime.datetime.strptime('2023-11-02 11:30:00', '%Y-%m-%d %H:%M:00')
    pv = pd.read_csv('test_pv(demand).csv', parse_dates=['date'])
    load = pd.read_csv('test_load(demand).csv', parse_dates=['date'])
    predWL = float(round(load.loc[load['date'] == test_date, 'load'].iloc[0], 2))
    predWPV = float(round(pv.loc[pv['date'] == test_date, 'pv'].iloc[0], 2))
    hour = int(load.loc[load['date'] == test_date, 'hour'].iloc[0])
    workday = int(load.loc[load['date'] == test_date, 'workday'].iloc[0])
    print('hour/workday : {}/{}, predL: {}, predPV: {}'.format(hour, workday, predWL, predWPV))

    soc = db_fn.get_pms_soc()
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL # 부하량

    #print('predL: {}, predPV: {}'.format(predWL, predWPV))
    #print('wcnd : {}, wsoc : {}({}%)'.format(round(wcnd,1), wsoc, soc))

    if 22 <= hour or hour < 8:  # 충전 22시 ~ 08시 (경부하)
        if cfg.max_capacity > wsoc : # soc 90% 초과 방지
            return min(cfg.max_capacity - wsoc, 150) # 경부하시간대 충전량 결정
        else:          # soc 90% 이상 충전 불가능
            return 0
    elif 11 == hour or 13 <= hour < 18:   # 방전 11~12시, 13~18시 (최대부하)
        if wcnd < 0: # 방전, 음수
            if cfg.min_capacity < wsoc:  # soc 10% 미만 확인
                return max(wcnd, -cfg.conv_capacity_1h, cfg.min_capacity - wsoc)  # soc 10% 까지 방전
            else:  # 방전량 만큼 방전
                return 0
        elif wcnd > 0: # 충전, 양수
            if cfg.max_capacity > wsoc :  # soc 90% 초과 방지
                return min(wcnd, 150, cfg.max_capacity - wsoc)  # 충전량 결정
            else:  # soc 90% 이상 충전 불가능
                return 0
        else:
            return 0
    else:   # 대기 08~11시, 12~13시, 18~22시 (중간부하)
        if wcnd > 0:  # 충전, 양수
            if cfg.max_capacity > wsoc:  # soc 90% 초과 방지
                return min(wcnd, 150, cfg.max_capacity - wsoc)  # 충전량 결정
            else:  # soc 90% 이상 충전 불가능
                return 0
        else:
            return 0


def pv_mode():
    '''
    # 예측된 부하량, 발전량 변수 가져오기
    pv_date = (datetime.datetime.now()).strftime('%Y-%m-%d %H:00:00')
    load_date = (datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))).strftime('%Y-%m-%d %H:%M:00')
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')
    predWL = float(round(load.loc[load['date'] == load_date, 'load'].iloc[0], 2))
    predWPV = float(round(pv.loc[pv['date'] == pv_date, 'pv'].iloc[0] / 4, 2))
    hour = datetime.datetime.now().hour
    '''

    # 예측된 부하량, 발전량 변수 가져오기
    test_date = (datetime.datetime.now() - timedelta(minutes=datetime.datetime.now().minute % 15)).strftime('%Y-%m-%d %H:%M:00')
    #test_date = datetime.datetime.strptime('2023-11-03 17:45:00', '%Y-%m-%d %H:%M:00')
    pv = pd.read_csv('test_pv(pv).csv', parse_dates=['date'])
    load = pd.read_csv('test_load(pv).csv', parse_dates=['date'])
    predWL = float(round(load.loc[load['date'] == test_date, 'load'].iloc[0], 2))
    predWPV = float(round(pv.loc[pv['date'] == test_date, 'pv'].iloc[0], 2))
    hour = int(load.loc[load['date'] == test_date, 'hour'].iloc[0])
    workday = int(load.loc[load['date'] == test_date, 'workday'].iloc[0])
    print('hour/workday : {}/{}, predL: {}, predPV: {}'.format(hour, workday, predWL, predWPV))

    soc = db_fn.get_pms_soc()
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL

    #print('predL: {}, predPV: {}'.format(predWL, predWPV))
    #print('wcnd : {}, wsoc : {}({}%)'.format(round(wcnd,1), wsoc, soc))


    if 10 <= hour < 16:  # 충전 10시 ~ 16시
        if predWPV > 0:
            if cfg.max_capacity > wsoc:  # soc 90% 초과 방지
                return min(predWPV, cfg.max_capacity - wsoc, cfg.conv_capacity_1h)
            else:
                return 0
        else :
            return 0
    else:     # 그외 나머지 시간 방전
        if wcnd < 0:  # 방전, 음수
            if cfg.min_capacity < wsoc:  # soc 10% 미만 확인
                return max(wcnd, -cfg.conv_capacity_1h, cfg.min_capacity - wsoc)  # soc 10% 까지 방전
            else:  # 방전량 만큼 방전
                return 0
        elif wcnd > 0:  # 충전, 양수
            if cfg.max_capacity > wsoc:  # soc 90% 초과 방지
                return min(cfg.max_capacity - wsoc, cfg.conv_capacity_1h, wcnd)  # 충전량 결정
            else:  # soc 90% 이상 충전 불가능
                return 0
        else:   # 방전 불가능 soc 10% 미만
            return 0

