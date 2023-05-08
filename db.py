import pandas as pd
# import mariadb
import sys
import pymysql

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

def get_pv_monitor():
    conn = conn_db()
    result = pd.DataFrame(columns=pv_colum)
    cur = conn.cursor()
    query = "SELECT * from pv_converter_monitoring"
    cur.execute(query)
    resultset = cur.fetchall()


    # sql Query 응답을 pd.DataFrame 으로 변경
    for id, date1, date2, date3, g_cur, g_mos1, g_mos2, g_power, g_volt, hw1, hw2, op, day, pv_cur, pv_mos1, pv_mos2, pv_num, pv_op, pv_power, pv_volt, sw1, sw2, dev_id in resultset:

        data = [id, date1, date2, date3, g_cur, g_mos1, g_mos2, g_power, g_volt, hw1, hw2, op, day, pv_cur, pv_mos1,
                pv_mos2, pv_num, pv_op, pv_power, pv_volt, sw1, sw2, dev_id]

        # Dataframe에 한행 씩 data 추가
        result.loc[id] = data

    conn.commit()
    conn.close()
    return result

def get_pqms_monitor():
    conn = conn_db()
    result = pd.DataFrame(columns=pqms_colum)
    cur = conn.cursor()
    query = "SELECT * from pqm_monitoring"
    cur.execute(query)
    resultset = cur.fetchall()

    for date, time_index, acdc, pv, essCharge, essDischarge, dcHome, interlink in resultset:
        data = [date, time_index, acdc, pv, essCharge, essDischarge, dcHome, interlink]

        result.loc[id] = data
    conn.commit()
    conn.close()
    return result

def get_ess_monitor():
    conn = conn_db()
    result = pd.DataFrame(columns=ess_colum)
    cur = conn.cursor()
    query = "SELECT * from ess_monitoring"
    cur.execute(query)
    resultset = cur.fetchall()

    for id,	created_date,	modified_date,	charging_capacity,	charging_status,	collected_date,	device_status,	discharge_capacity,	ess_current,	ess_power,	ess_power_available,	ess_voltage,	grid_connectivity,	grid_current,	grid_power,	grid_voltage,	internal_connectivity,	max_soc,	min_soc,	negative_electrolyte_temperature,	number_of_abnormal_stack,	number_of_normal_stack,	operating_pump,	operation_state,	positive_electrolyte_temperature,	pump_speed,	soc,	temperature, voltage_current_mode,	device_id in resultset:
        data = [id,	created_date,	modified_date,	charging_capacity,	charging_status,	collected_date,	device_status,	discharge_capacity,	ess_current,	ess_power,	ess_power_available,	ess_voltage,	grid_connectivity,	grid_current,	grid_power,	grid_voltage,	internal_connectivity,	max_soc,	min_soc,	negative_electrolyte_temperature,	number_of_abnormal_stack,	number_of_normal_stack,	operating_pump,	operation_state,	positive_electrolyte_temperature,	pump_speed,	soc,	temperature, voltage_current_mode,	device_id]

        result.loc[id] = data
    conn.commit()
    conn.close()
    return result





