import db
import datetime
import mqtt_fn
import config as cfg

now = datetime.datetime.now()
now_date = datetime.datetime.now().strftime('%Y-%m-%d')
now_hour = datetime.datetime.now().strftime('%H:%M:%S')


def predict_load():
    # 날짜 포맷 변경 및 과거 데이터 기간 설정
    past = now - datetime.timedelta(days=90)
    past = past.strftime('%Y-%m-%d')

    pred_load = db.get_pv_monitor(past, now_date)
    print(pred_load)

    pred_load.to_csv('test.csv')
    mqtt_fn.mqttc.publish(cfg.pub_pms_topic, f'get?p_index={mqtt_fn.pms_index}&soc_report')
    mqtt_fn.pms_index+=1





    return pred_load

def predict_pv():
    past = now - datetime.timedelta(days=90)
    past = past.strftime('%Y-%m-%d')

    pred_pv = db.get_pv_monitor(past, now_date)
    # 현재값 불러올때 pv는 데이터를
    # get?p_index&pqms_load


    return pred_pv

def data_preprocessing():
    return

def calculate_pf(limit=None, time=None, pf=None):

    if (limit!=None):
        predict_load()
    elif(time!=None):
        print(time)
    elif(pf!=None):
        print(pf)
    else:
        print()

    pf = 12.1 # test pf

    return pf

def optimize_mode():
    print("optimize_mode")
    pf = calculate_pf()
    return pf

def peak_mode(limit):
    print(f'peak_mode, limit= {limit}')
    pf = calculate_pf(limit)
    return pf

def demand_mode():
    print("demand_mode")
    pf = calculate_pf()
    return pf

def pv_mode():
    print("pv_mode")
    pf = calculate_pf()
    return pf
