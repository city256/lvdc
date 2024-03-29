import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import sys
from pytimekr import pytimekr
import pymysql
import config as cfg

def conn_db():
    try:
        conn = pymysql.connect(
            user=cfg.db_user,
            password=cfg.db_pw,
            host=cfg.broker_url,
            port=cfg.db_port,
            db=cfg.db_name
        )
    except pymysql.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
    return conn

def put_chaged_date(origin_date, change_date):
    conn = conn_db()
    cur = conn.cursor()

    query = f"""
    INSERT INTO pqms_load_min_test (id, created_date, modified_date, p_time, ac_dc, dcdc, dc_home, ess_charge, ess_discharge, interlink, load_date, p_error, p_id, p_info, p_type, pqms_index, pv, time_index)
            SELECT 
                NULL,
                DATE_ADD(created_date, INTERVAL {(datetime.datetime.strptime(change_date, '%Y-%m-%d') - datetime.datetime.strptime(origin_date, '%Y-%m-%d')).days} DAY),
                DATE_ADD(modified_date, INTERVAL {(datetime.datetime.strptime(change_date, '%Y-%m-%d') - datetime.datetime.strptime(origin_date, '%Y-%m-%d')).days} DAY),
                DATE_ADD(p_time, INTERVAL {(datetime.datetime.strptime(change_date, '%Y-%m-%d') - datetime.datetime.strptime(origin_date, '%Y-%m-%d')).days} DAY),
                ac_dc,
                dcdc,
                dc_home,
                ess_charge,
                ess_discharge,
                interlink,
                DATE_ADD(load_date, INTERVAL {(datetime.datetime.strptime(change_date, '%Y-%m-%d') - datetime.datetime.strptime(origin_date, '%Y-%m-%d')).days} DAY),
                p_error,
                p_id,
                p_info,
                p_type,
                pqms_index,
                pv,
                time_index
            FROM 
                pqms_load_min_test
            WHERE 
                load_date BETWEEN '{origin_date} 00:00:00' AND '{origin_date} 23:59:59';
    """
    cur.execute(query)
    conn.commit()
    conn.close()
    return

#put_chaged_date('2023-09-18', '2023-10-16')

def put_pqms_15():
    conn = conn_db()
    cur = conn.cursor()

    query = """
    INSERT INTO pqms_load_min_test (id, created_date, modified_date, p_time, ac_dc, dcdc, dc_home, ess_charge, ess_discharge, interlink, load_date, p_error, p_id, p_info, p_type, pqms_index, pv, time_index)
    SELECT 
        id,
        created_date, 
        modified_date, 
        p_time,
        ac_dc - LAG(ac_dc, 1) OVER (ORDER BY load_date) AS ac_dc,
        dcdc - LAG(dcdc, 1) OVER (ORDER BY load_date) AS dcdc,
        dc_home - LAG(dc_home, 1) OVER (ORDER BY load_date) AS dc_home,
        ess_charge - LAG(ess_charge, 1) OVER (ORDER BY load_date) AS ess_charge,
        ess_discharge - LAG(ess_discharge, 1) OVER (ORDER BY load_date) AS ess_discharge,
        interlink - LAG(interlink, 1) OVER (ORDER BY load_date) AS interlink,
        load_date,
        p_error,
        p_id,
        p_info,
        p_type,
        pqms_index,
        pv - LAG(pv, 1) OVER (ORDER BY load_date) AS pv,
        time_index
    FROM 
        pqms_load_event
    WHERE load_date BETWEEN '2023-08-27' AND '2023-09-26'
    ON DUPLICATE KEY UPDATE load_date
    """

    cur.execute(query)
    conn.commit()
    conn.close()
    return

def put_pqms_data(json):
    conn = conn_db()
    cur = conn.cursor()
    # id, create_date, modified_date, ac_dc, dcdc, dc_home, ess_charge, ess_discharge, interlink, load_date, p_error, p_id, p_info, p_time, p_type, pqms_index, pv, time_index
    date = datetime.datetime.strptime(json['p_time'], '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
    p_type = json['p_type']
    p_info = json['p_info']
    p_index = json['p_index']
    p_error = json['p_error']
    p_id = json['p_id']
    index = json['p_contents']['index']
    acdc = json['p_contents']['ACDC']
    pv = json['p_contents']['PV']
    dcdc = json['p_contents']['DCDC']
    esscharge = json['p_contents']['ESSCharge']
    essdischarge = json['p_contents']['ESSDischarge']
    dchome = json['p_contents']['DCHome']
    interlink = json['p_contents']['Interlink']

    query = """
        INSERT INTO pqms_load_event (id, created_date, modified_date, ac_dc, dcdc, dc_home, ess_charge, ess_discharge, interlink, load_date, p_error, p_id, p_info, p_time, p_type, pqms_index, pv, time_index)
        VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE load_date=%s;
    """
    data = (date, date, acdc, dcdc, dchome, esscharge, essdischarge, interlink, date, p_error, p_id, p_info, date, p_type, p_index, pv, index, date)
    cur.execute(query, data)
    conn.commit()
    conn.close()
    return

def put_pqms_data_index(time, index, acdc, dcdc, dchome, interlink, pv):
    conn = conn_db()
    cur = conn.cursor()
    # id, create_date, modified_date, ac_dc, dcdc, dc_home, ess_charge, ess_discharge, interlink, load_date, p_error, p_id, p_info, p_time, p_type, pqms_index, pv, time_index
    date = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    p_type = 'get'
    p_info = 'pqms_load'
    p_index = index
    p_error = 0
    p_id = 'MG1_SVR_PQMS'
    index = index
    acdc = acdc
    dcdc = dcdc
    dchome = dchome
    interlink = interlink
    pv = pv
    esscharge = 0
    essdischarge = 0

    query = """
        INSERT INTO pqms_load_event (id, created_date, modified_date, ac_dc, dcdc, dc_home, ess_charge, ess_discharge, interlink, load_date, p_error, p_id, p_info, p_time, p_type, pqms_index, pv, time_index)
        VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE load_date=%s;
    """
    data = (date, date, acdc, dcdc, dchome, esscharge, essdischarge, interlink, date, p_error, p_id, p_info, date, p_type, p_index, pv, index, date)
    cur.execute(query, data)
    conn.commit()
    conn.close()
    return

def get_pqms_data_hour():
    conn = conn_db()
    cur = conn.cursor()
    query = """
    SELECT 
        CASE 
            WHEN MINUTE(load_date) = 0 THEN DATE_FORMAT(DATE_SUB(load_date, INTERVAL 1 HOUR), '%Y-%m-%d %H:00:00')
            ELSE DATE_FORMAT(load_date, '%Y-%m-%d %H:00:00') 
        END AS target_hour,
        ROUND(SUM(ac_dc), 2),
        ROUND(SUM(dcdc + interlink + dc_home), 2) as total_value,
        ROUND(SUM(pv), 2),
        ROUND(SUM(ess_charge), 2),
        ROUND(SUM(ess_discharge), 2),
        CASE
            WHEN WEEKDAY(load_date) IN (5,6) THEN 0
            WHEN load_date IN ('2023-08-15','2023-09-28','2023-09-29','2023-09-30','2023-10-02','2023-10-03','2023-10-09', '2023-12-25') THEN 0
            ELSE 1
        END as workday
    FROM pqms_load_min
    WHERE MINUTE(load_date) >= 15 OR MINUTE(load_date) = 0
    GROUP BY target_hour
    ORDER BY target_hour
"""
    cur.execute(query)
    resultset = cur.fetchall()
    result = pd.DataFrame(resultset, columns=['date', 'acdc', 'load', 'pv', 'ess_charge', 'ess_discharge', 'workday'])
    result.to_csv('pqms_data.csv')
    conn.commit()
    conn.close()
    return result

def get_pqms_data_15():
    conn = conn_db()
    cur = conn.cursor()
    query = """
    SELECT 
        DATE_FORMAT(load_date, '%Y-%m-%d %H:%i:00'),
        ROUND((ac_dc), 2),
        ROUND((dcdc + interlink + dc_home), 2) as total_value,
        ROUND((pv), 2),
        ROUND((ess_charge), 2),
        ROUND((ess_discharge), 2),
        CASE
            WHEN WEEKDAY(load_date) IN (5,6) THEN 0
            WHEN load_date IN ('2023-08-15','2023-09-28','2023-09-29','2023-09-30','2023-10-02','2023-10-03','2023-10-09', '2023-12-25') THEN 0
            ELSE 1
        END as workday
    FROM pqms_load_min
"""
    cur.execute(query)
    resultset = cur.fetchall()
    result = pd.DataFrame(resultset, columns=['date', 'acdc', 'load', 'pv', 'ess_charge', 'ess_discharge', 'workday'])
    result.to_csv('pqms_data_pv.csv')
    conn.commit()
    conn.close()
    return result
#get_pqms_data_15()

def get_pqms_data_origin():
    conn = conn_db()
    cur = conn.cursor()
    query = """
    SELECT 
        DATE_FORMAT(load_date, '%Y-%m-%d %H:%i:00'),
        ROUND((ac_dc - ess_discharge), 2),
        ROUND((dcdc + interlink + dc_home), 2) as total_value,
        ROUND((pv), 2),
        ROUND((ess_charge), 2),
        ROUND((ess_discharge), 2),
        CASE
            WHEN WEEKDAY(load_date) IN (5,6) THEN 0
            WHEN load_date IN ('2023-08-15','2023-09-28','2023-09-29','2023-09-30','2023-10-02','2023-10-03','2023-10-09', '2023-12-25') THEN 0
            ELSE 1
        END as workday
    FROM pqms_load_min
"""
    cur.execute(query)
    resultset = cur.fetchall()
    result = pd.DataFrame(resultset, columns=['date', 'acdc', 'load', 'pv', 'ess_charge', 'ess_discharge', 'workday'])
    result.to_csv('pqms_data_origin.csv')
    conn.commit()
    conn.close()
    return result

def check_date(date_str):

    date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:00')
    # 해당 날짜가 공휴일인지 확인
    holidays = pytimekr.holidays(date.year)

    if date.weekday() >= 5 or date.date() in holidays:
        return 0  # 공휴일 또는 주말
    else:
        return 1  # 평일, 주중

def get_load_data():
    conn = conn_db()
    cur = conn.cursor()
    query = """
    SELECT 
        CASE 
            WHEN MINUTE(load_date) = 0 THEN DATE_FORMAT(DATE_SUB(load_date, INTERVAL 1 HOUR), '%Y-%m-%d %H:00:00')
            ELSE DATE_FORMAT(load_date, '%Y-%m-%d %H:00:00') 
        END AS target_hour,
        SUM(dcdc + interlink + dc_home) as total_value
    FROM pqms_load_min_test
    WHERE MINUTE(load_date) >= 15 OR MINUTE(load_date) = 0
    GROUP BY target_hour
    ORDER BY target_hour
"""
    cur.execute(query)
    resultset = cur.fetchall()
    result = pd.DataFrame(resultset, columns=['date', 'load'])

    # 날짜별 근무일 항목 생성
    result['workday'] = result['date'].apply(check_date)

    # 다음 24시간 날짜 데이터 생성
    next_24_hours_dates = pd.date_range(start=(datetime.datetime.now() ).strftime('%Y-%m-%d %H:00:00'), periods=24, freq='H')

    # load는 null, workday는 1로 설정
    new_data = {'date': next_24_hours_dates,
                'load': [np.nan] * 24,
                'workday': [check_date(str(date)) for date in next_24_hours_dates]}

    # 새로운 DataFrame 생성
    new_df = pd.DataFrame(new_data)

    # 원본 DataFrame에 새로운 데이터 추가
    result = pd.concat([result, new_df], ignore_index=True)
    result.to_csv('load_data.csv')
    conn.commit()
    conn.close()
    return result


def get_load_data_15():
    conn = conn_db()
    cur = conn.cursor()
    query = """
    SELECT 
        DATE_FORMAT(load_date, '%Y-%m-%d %H:%i:00'),
        ROUND((dcdc + interlink + dc_home), 2) as total_value
    FROM pqms_load_min_test
"""
    cur.execute(query)
    resultset = cur.fetchall()
    result = pd.DataFrame(resultset, columns=['date', 'load'])

    # 날짜별 근무일 항목 생성
    result['workday'] = result['date'].apply(check_date)

    # 다음 15분 주기 계산
    next_15_minutes = datetime.datetime.now() + timedelta(minutes=(15 - datetime.datetime.now().minute % 15))

    # 다음 24시간의 15분 데이터 생성
    next_24_hours_dates = pd.date_range(start=next_15_minutes.strftime('%Y-%m-%d %H:%M:00'), periods=96, freq='15T')

    # load는 null, workday는 1로 설정
    new_data = {'date': next_24_hours_dates,
                'load': [np.nan] * 96,
                'workday': [check_date(str(date)) for date in next_24_hours_dates]}
    # 새로운 DataFrame 생성
    next_24 = pd.DataFrame(new_data)

    # 원본 DataFrame에 새로운 데이터 추가
    result = pd.concat([result, next_24], ignore_index=True)
    conn.commit()
    conn.close()
    result.to_csv('load_data_15.csv')
    return result


def get_pv_data():   # DB에서 PQMS의 PV 발전량만 가져옴
    conn = conn_db()
    cur = conn.cursor()
    query = """
    SELECT 
        CASE 
            WHEN MINUTE(load_date) = 0 THEN DATE_FORMAT(DATE_SUB(load_date, INTERVAL 1 HOUR), '%Y-%m-%d %H:00:00')
            ELSE DATE_FORMAT(load_date, '%Y-%m-%d %H:00:00') 
        END AS pv_date,
        ROUND(SUM(pv), 2) as pv
    FROM pqms_load_min_test
    WHERE MINUTE(load_date) >= 15 OR MINUTE(load_date) = 0
    GROUP BY pv_date
    ORDER BY pv_date
    """
    cur.execute(query)
    resultset = cur.fetchall()
    result = pd.DataFrame(resultset, columns=['date', 'pv'])
    conn.commit()
    conn.close()
    return result

def get_weather_data():   # DB에서 PMS의 일사량, 온도 가져옴
    conn = conn_db()
    cur = conn.cursor()
    query = """
    SELECT 
		DATE_FORMAT(created_date, '%Y-%m-%d %H:00:00') AS weather_date,
        ROUND(AVG(weather_sunlight),2) as avg_sunlight,
        ROUND(AVG(weather_temperature),2) as avg_temp
    FROM pms_converter_get
    GROUP BY weather_date
    ORDER BY weather_date
    """
    cur.execute(query)
    resultset = cur.fetchall()

    result = pd.DataFrame(resultset, columns=['date', 'sunlight','temperature'])

    conn.commit()
    conn.close()
    return result

def get_sunlight_data():  #DB에서 PMS의 일사량만 가져옴
    conn = conn_db()
    cur = conn.cursor()
    query = """
    SELECT 
		DATE_FORMAT(created_date, '%Y-%m-%d %H:00:00') AS weather_date,
        ROUND(AVG(weather_sunlight),2) as avg_sunlight
    FROM pms_converter_get
    GROUP BY weather_date
    ORDER BY weather_date
    """
    cur.execute(query)
    resultset = cur.fetchall()

    result = pd.DataFrame(resultset, columns=['date', 'sunlight'])

    conn.commit()
    conn.close()
    return result


def get_pv_dataset():   # PQMS의 PV발전량과 PMS의 날씨 데이터 병합
    pv = get_pv_data()
    weather = get_weather_data()
    pv['date'] = pd.to_datetime(pv['date'], format='%Y-%m-%d %H:00:00')
    pv.set_index('date')
    weather['date'] = pd.to_datetime(weather['date'], format='%Y-%m-%d %H:00:00')
    weather.set_index('date')
    merge = pd.merge(weather, pv)
    merge.to_csv('pv_db_data.csv')
    return merge

def get_pms_soc():
    conn = conn_db()
    cur = conn.cursor()
    query = "SELECT current_soc from pms_converter_get ORDER BY id DESC LIMIT 1"
    cur.execute(query)
    resultset = cur.fetchall()

    conn.commit()
    conn.close()
    return resultset[0][0]

