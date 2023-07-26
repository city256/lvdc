import pandas as pd
import sys
import pymysql
import datetime

# maria db config
db_user = "root"
db_pw = "Lvdc12341@"
db_name = "etri_lvdc"
db_port = 13306
url = 'lvdc.iptime.org'


pv_colum = ['id', 'date', 'date', 'date', 'gcur', 'gmos1', 'gmos2', 'gpower', 'gvolt',
                                          'h1', 'h2', 'op', 'day', 'pvcur', 'pvmos1', 'pvmos2', 'pvnum', 'pvop',
                                          'pvpower', 'pvvolt', 's1', 's2', 'devid']
ess_colum = ['id', 'created_date', 'modified_date', 'charging_capacity', 'charging_status', 'collected_date' ,'device_status',	'discharge_capacity',	'ess_current',	'ess_power',	'ess_power_available',	'ess_voltage',	'grid_connectivity',	'grid_current,'	'grid_power',	'grid_voltage','internal_connectivity',	'max_soc',	'min_soc',	'negative_electrolyte_temperature,'	'number_of_abnormal_stack',	'number_of_normal_stack',	'operating_pump',	'operation_state',	'positive_electrolyte_temperature',	'pump_speed',	'soc',	'temperature',	'voltage_current_mode',	'device_id']
pqms_colum = ['date', 'time_index','acdc','pv','essCharge','essDischarge','dcHome','interlink']


def conn_db():
    try:
        conn = pymysql.connect(
            user=db_user,
            password=db_pw,
            host=url,
            port=db_port,
            db=db_name
        )
        print("connection Success!")
    except pymysql.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
    return conn

def get_test():
    conn = conn_db()
    cur = conn.cursor()
    query = 'SELECT * FROM test WHERE date BETWEEN DATE_ADD(NOW(), INTERVAL -6 MONTH) AND NOW();'
    cur.execute(query)
    resultset = cur.fetchall()

    result = pd.DataFrame(resultset, columns=['date', 'load'])
    conn.commit()
    conn.close()
    return result

get_test()
def get_pv_monitor(past):
    conn = conn_db()
    cur = conn.cursor()

    # past 부터 now 까지의 데이터 쿼리
    query = f'select * from pv_converter_monitoring where collected_date between \'{past}\' and NOW() '

    cur.execute(query)
    resultset = cur.fetchall()

    result = pd.DataFrame(resultset, columns=pv_colum)
    conn.commit()
    conn.close()
    return result

def get_pqms_monitor():
    conn = conn_db()
    cur = conn.cursor()
    query = "SELECT * from pqm_monitoring"
    cur.execute(query)
    resultset = cur.fetchall()

    result = pd.DataFrame(resultset, columns=pqms_colum)
    conn.commit()
    conn.close()
    return result

def get_ess_monitor():
    conn = conn_db()
    cur = conn.cursor()
    query = "SELECT * from ess_monitoring"
    cur.execute(query)
    resultset = cur.fetchall()

    result = pd.DataFrame(resultset, columns=ess_colum)

    conn.commit()
    conn.close()
    return result


