import db
import datetime
now = datetime.datetime.now()
now_date = datetime.datetime.now().strftime('%Y-%m-%d')
now_hour = datetime.datetime.now().strftime('%H:%M:%S')


def predict_load():
    # 날짜 포맷 변경 및 과거 데이터 기간 설정
    past = now - datetime.timedelta(days=90)
    past = past.strftime('%Y-%m-%d')

    pv_df = db.get_pv_monitor(past, now_date)
    print(pv_df)

    pv_df.to_csv('test.csv')


    return

def predict_pv():

    return

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

    pf = 12.1

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
