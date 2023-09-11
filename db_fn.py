import pandas as pd
import sys
import pymysql
import config as cfg


pv_colum = ['id', 'date', 'date', 'date', 'gcur', 'gmos1', 'gmos2', 'gpower', 'gvolt',
                                          'h1', 'h2', 'op', 'day', 'pvcur', 'pvmos1', 'pvmos2', 'pvnum', 'pvop',
                                          'pvpower', 'pvvolt', 's1', 's2', 'devid']
ess_colum = ['id', 'created_date', 'modified_date', 'charging_capacity', 'charging_status', 'collected_date' ,'device_status',	'discharge_capacity',	'ess_current',	'ess_power',	'ess_power_available',	'ess_voltage',	'grid_connectivity',	'grid_current,'	'grid_power',	'grid_voltage','internal_connectivity',	'max_soc',	'min_soc',	'negative_electrolyte_temperature,'	'number_of_abnormal_stack',	'number_of_normal_stack',	'operating_pump',	'operation_state',	'positive_electrolyte_temperature',	'pump_speed',	'soc',	'temperature',	'voltage_current_mode',	'device_id']
pqms_colum = ['id', 'create_date','modified_date','ac_dc','dcdc','dc_home','ess_charge','ess_discharge', 'interlink', 'load_date', 'p_error', 'p_id','p_info','p_time','p_type','pqms_index','pv','time_index']



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
        SUM(dcdc + interlink + dc_home) as total_value
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
def get_pv_data():
    conn = conn_db()
    cur = conn.cursor()
    query = """
    SELECT 
        CASE 
            WHEN MINUTE(load_date) = 0 THEN DATE_FORMAT(DATE_SUB(load_date, INTERVAL 1 HOUR), '%Y-%m-%d %H:00:00')
            ELSE DATE_FORMAT(load_date, '%Y-%m-%d %H:00:00') 
        END AS pv_date,
        SUM(pv) as pv
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



def get_weather_data():
    conn = conn_db()
    cur = conn.cursor()
    query = """
    SELECT 
		DATE_FORMAT(created_date, '%Y-%m-%d %H:00:00') AS weather_date,
        AVG(weather_sunlight) as avg_sunlight,
        AVG(weather_temperature) as avg_sunlight
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

def get_pv_dataset():
    pv = get_pv_data()
    weather = get_weather_data()
    pv['date'] = pd.to_datetime(pv['date'], format='%Y-%m-%d %H:00:00')
    pv.set_index('date')
    weather['date'] = pd.to_datetime(weather['date'], format='%Y-%m-%d %H:00:00')
    weather.set_index('date')
    merge= pd.merge(weather, pv)
    print(merge)
    return merge

get_pv_dataset()

def get_pms_soc():
    conn = conn_db()
    cur = conn.cursor()
    query = "SELECT current_soc from pms_converter_get ORDER BY id DESC LIMIT 1"
    cur.execute(query)
    resultset = cur.fetchall()

    conn.commit()
    conn.close()
    return resultset[0][0]
