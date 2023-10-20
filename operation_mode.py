import config as cfg
import datetime
import pandas as pd
import db_fn

def optimize_mode():
    # 예측된 부하량, 발전량 변수 가져오기
    date_str = (datetime.datetime.now()).strftime('%Y-%m-%d %H:00:00')
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')
    predWL = float(round(load.loc[load['date'] == date_str, 'load'].iloc[0], 2))
    predWPV = float(round(pv.loc[pv['date'] == date_str, 'pv'].iloc[0], 2))
    hour = datetime.datetime.now().hour

    soc = db_fn.get_pms_soc()
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL

    #print('predL: {}, predPV: {}, soc: {}'.format(predWL, predWPV, soc))
    #print('wcnd : {}, wsoc : {}({}%)'.format(round(wcnd,1), wsoc, soc))

    if wcnd < 0: # 방전, 음수
        if 8 < hour < 19 and db_fn.check_date(datetime.datetime.now().date()):  # 근무일, 근무시간 9~18시 확인
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


def peak_mode(limit):
    # 예측된 부하량, 발전량 변수 가져오기
    date_str = (datetime.datetime.now()).strftime('%Y-%m-%d %H:00:00')
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')
    predWL = float(round(load.loc[load['date'] == date_str, 'load'].iloc[0], 2))
    predWPV = float(round(pv.loc[pv['date'] == date_str, 'pv'].iloc[0], 2))
    hour = datetime.datetime.now().hour

    soc = db_fn.get_pms_soc()
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL
    # peak_limit = limit # 함수 변수 사용
    peak_limit = cfg.peak_limit

    if 22 <= hour or hour < 8:  # 충전 22시 ~ 08시
        if cfg.max_capacity > wsoc:    # soc 90% 초과 방지
            return min(cfg.max_capacity - wsoc, cfg.conv_capacity_1h)  # 최대 출력으로 충전
        else:  # 충전 불가능
            return 0
    else:       # 방전 09시 ~ 21시
        if -wcnd > peak_limit:   # 피크 초과시
            if wsoc >= cfg.min_capacity: # 배터리 잔여 전력 확인
                if cfg.min_capacity > wsoc + wcnd + peak_limit:   # soc 10% 미만 확인
                    return cfg.min_capacity - wsoc # 10% 까지만 방전
                else:   # 피크치 만큼 방전
                    return max(wcnd + peak_limit, -cfg.conv_capacity_1h) # 피크치까지 방전
            else:   # ESS에 잔여 전력 없을시
                return 0
        else:   # 피크 초과 안할시
            return 0



def demand_mode():
    # 예측된 부하량, 발전량 변수 가져오기
    date_str = (datetime.datetime.now()).strftime('%Y-%m-%d %H:00:00')
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')
    predWL = float(round(load.loc[load['date'] == date_str, 'load'].iloc[0], 2))
    predWPV = float(round(pv.loc[pv['date'] == date_str, 'pv'].iloc[0], 2))
    hour = datetime.datetime.now().hour

    soc = db_fn.get_pms_soc()
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL # 부하량

    #print('predL: {}, predPV: {}'.format(predWL, predWPV))
    #print('wcnd : {}, wsoc : {}({}%)'.format(round(wcnd,1), wsoc, soc))

    if 22 <= hour or hour < 8:  # 충전 22시 ~ 08시 (경부하)
        if cfg.max_capacity > wsoc : # soc 90% 초과 방지
            return min(cfg.max_capacity - wsoc, cfg.conv_capacity_1h) # 경부하시간대 충전량 결정
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
                return min(cfg.max_capacity - wsoc, cfg.conv_capacity_1h, wcnd)  # 충전량 결정
            else:  # soc 90% 이상 충전 불가능
                return 0
        else:
            return 0
    else:   # 대기 08~11시, 12~13시, 18~22시 (중간부하)
        return 0


def pv_mode():
    # 예측된 부하량, 발전량 변수 가져오기
    date_str = (datetime.datetime.now()).strftime('%Y-%m-%d %H:00:00')
    pv = pd.read_csv('pred_pv.csv')
    load = pd.read_csv('pred_load.csv')
    predWL = float(round(load.loc[load['date'] == date_str, 'load'].iloc[0], 2))
    predWPV = float(round(pv.loc[pv['date'] == date_str, 'pv'].iloc[0], 2))

    soc = db_fn.get_pms_soc()
    wsoc = cfg.ess_capacity * (soc * 0.01)
    wcnd = predWPV - predWL
    hour = datetime.datetime.now().hour

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
