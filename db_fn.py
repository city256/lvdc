import pandas as pd
import sys
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
        print("DB connection Success!")
    except pymysql.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
    return conn

def get_test():
    conn = conn_db()
    cur = conn.cursor()
    query = 'SELECT * FROM test WHERE date BETWEEN DATE_ADD(NOW(), INTERVAL -6 MONTH) AND NOW() ORDER BY date ASC LIMIT 500;'
    #query = "SELECT current_soc from pms_converter_get ORDER BY id DESC LIMIT 6"

    cur.execute(query)
    resultset = cur.fetchall()

    result = pd.DataFrame(resultset, columns=['date', 'load'])
    conn.commit()
    conn.close()
    return result

def get_load_data():
    conn = conn_db()
    cur = conn.cursor()
    query = """
    SELECT 
        CASE 
            WHEN MINUTE(load_date) = 0 THEN DATE_FORMAT(DATE_SUB(load_date, INTERVAL 1 HOUR), '%Y-%m-%d %H:00:00')
            ELSE DATE_FORMAT(load_date, '%Y-%m-%d %H:00:00') 
        END AS target_hour,
        ROUND(SUM(dcdc + interlink + dc_home),2) as total_value
    FROM pqms_load_min
    WHERE MINUTE(load_date) >= 15 OR MINUTE(load_date) = 0
    GROUP BY target_hour
    ORDER BY target_hour
"""
    cur.execute(query)
    resultset = cur.fetchall()
    result = pd.DataFrame(resultset, columns=['date', 'load'])

    conn.commit()
    conn.close()
    result.to_csv('load_db_data.csv')
    return result

#print(get_pqms_data())

'''print(pqms_data['time_index'])
print(len(pqms_data))
i=0
for i in range(len(pqms_data)):
    if pqms_data['time_index'][i]%15==0:
        index = pqms_data['time_index'][i]/15
        if(index%4==0):
            print(pqms_data['time_index'][i], index)
            
        #print(pqms_data.loc['time_index'][0], index)
'''
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
    FROM pqms_load_min
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
print(get_load_data())


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
