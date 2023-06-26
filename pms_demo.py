#!/usr/bin/env python3
import datetime
import json
import paho.mqtt.client as mqttc
import random

# mqtt config
url = 'lvdc.iptime.org'
sub_topic = 'lvdc/pms'
pub_ais_topic = 'lvdc/ais'
pub_ems_topic = 'lvdc/ems/pms'
mqtt_port = 1883

# global parameters
p_index=0
operation_state = None

# event_msg
def event_operation_state():
    event={
        'p_type':'event',
        'p_id':'MG1_SVR_PMS',
        'p_cmd':'event/operation_state',
        'p_time':'{}'.format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
        'p_contents':{
            'operation_state':operation_state
        }
    }
    json_string = json.dumps(event)
    return json_string
def event_converter_fault(color):
    pms_state = 'normal'
    if color == 'red':
        pms_state = 'fault'
    elif color == 'green':
        pms_state = 'normal'
    event = {
        'p_type': 'event',
        'p_id': 'MG1_SVR_PMS',
        'p_cmd': 'event/battery_fault',
        'p_time': '{}'.format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
        'p_contents': {
            'pms_state': pms_state,
            'communication': color,
            'grid_OV': color,
            'grid_UV': color,
            'grid_OC': color,
            'batt_OV': color,
            'batt_UV': color,
            'batt_OC': color,
            'temperature': color,
            'hw': color,
            'short_circuit': color
        }
    }
    json_string = json.dumps(event)
    return json_string
def event_battery_fault(color):
    batt_state = 'normal'
    if color == 'red':
        batt_state = 'fault'
    elif color == 'green':
        batt_state = 'normal'
    event={
        'p_type':'event',
        'p_id':'MG1_SVR_PMS',
        'p_cmd':'event/battery_fault',
        'p_time':'{}'.format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
        'p_contents':{
            'battery_state':batt_state,
            'communication':color,
            'stack_OV':color,
            'stack_UV':color,
            'pipe_OP': color,
            'pipe_UP':color,
            'pipe_PRS':color,
            'tank_exceed':color,
            'tank_fault': color,
            'temperature':color,
            'batt_leak':color
        }
    }
    json_string = json.dumps(event)
    return json_string

# get response msg
def response_soc_report(p_index):
    response = {
        'p_type': 'get',
        'p_id': 'MG1_SVR_PMS',
        'p_cmd': 'response/soc_report',
        'p_index': p_index,
        'p_error': 0,
        'p_time': '{}'.format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
        'p_contents': {
            'min_soc': 10,
            'max_soc': 90,
            'current_soc': round(random.uniform(11, 90), 1),
            'weather_temperature': round(random.uniform(0, 40), 1),
            'weather_humidity': round(random.uniform(0, 40), 1),
            'weather_sunlight': round(random.uniform(0, 40), 1)
        }
    }
    json_string = json.dumps(response)
    return json_string
def response_converter_info(p_index):
    response = {
        'p_type': 'get',
        'p_id': 'MG1_SVR_PMS',
        'p_cmd': 'response/converter_info',
        'p_index': p_index,
        'p_error': 0,
        'p_time': '{}'.format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
        'p_contents': {
            'operation_state':operation_state,
            'input_voltage':round(random.uniform(100, 250), 1),
            'input_power':round(random.uniform(100, 250), 1),
            'output_voltage':round(random.uniform(100, 250), 1),
            'output_power':round(random.uniform(100, 250), 1),
            'min_soc': 10,
            'max_soc': 90,
            'current_soc': round(random.uniform(11, 90), 1),
            'weather_temperature': round(random.uniform(0, 40), 1),
            'weather_humidity': round(random.uniform(0, 40), 1),
            'weather_sunlight': round(random.uniform(0, 40), 1)
        }
    }
    json_string = json.dumps(response)
    return json_string

# set response msg
def response_operation_state(p_index, state):
    response = {
        'p_type': 'set',
        'p_id': 'MG1_SVR_PMS',
        'p_cmd': 'response/operation_state',
        'p_index': p_index,
        'p_error': 0,
        'p_time': '{}'.format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
        'p_contents': {
            'operation_state': state
        }
    }
    json_string = json.dumps(response)
    return json_string
def response_operation_mode(p_index, mode, pref=None):
    if pref == None:
        pref = round(random.uniform(-250, 250), 1)
    response = {
        'p_type': 'set',
        'p_id': 'MG1_SVR_PMS',
        'p_cmd': 'response/operation_mode',
        'p_index': p_index,
        'p_error': 0,
        'p_time': '{}'.format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
        'p_contents': {
            'operation_mode': mode,
            'power_reference': pref
        }
    }
    json_string = json.dumps(response)
    return json_string

def on_connect(client, userdata, flags, rc):
    # print("rc: " + str(rc))
    if rc==0:
        print("connected OK Returned code=",rc)
    else:
        print("Bad connection Returned code=",rc)

# 브로커에게 메시지가 도착하면 on_message 실행 (이벤트가 발생하면 호출)
def on_message(client, obj, msg):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
    msg_handler(msg.payload)

def on_publish(client, obj, mid):
    # 용도 : publish를 보내고 난 후 처리를 하고 싶을 때
    # 사실 이 콜백함수는 잘 쓰진 않는다.
    #print("Publish Complete: " + str(mid))
    print()
def on_subscribe(client, obj, mid, granted_qos):
    #print("Subscribe Complete : " + str(mid) + " " + str(granted_qos))
    print()
def msg_handler(msg):
    msg_str = str(msg)[2:-1] # payload 문자열처리

    # get 명령어 처리
    if msg_str[0:3]== "get":
        msg_str=msg_str[4:]
        split_msg = msg_str.split("&")
        recv_index = int(split_msg[0].split('=')[1])

        if split_msg[1].split('=')[0]== 'converter_info': # converter_info msg
            pub_msg = response_converter_info(recv_index)
            mqtt.publish(pub_ems_topic, pub_msg)
        elif split_msg[1].split('=')[0]== 'soc_report': # soc_report msg
            pub_msg = response_soc_report(recv_index)
            mqtt.publish(pub_ais_topic, pub_msg)
    # set 명령어 처리
    elif msg_str[0:3]== "set":
        msg_str = msg_str[4:]
        split_msg = msg_str.split("&")
        recv_index = int(split_msg[0].split('=')[1])
        if split_msg[1].split('=')[0]== 'operation_state': # operation_state msg
            global operation_state
            operation_state = int(split_msg[1].split('=')[1])
            pub_msg = response_operation_state(recv_index, operation_state)
            mqtt.publish(pub_ais_topic, pub_msg)
            pub_msg = event_operation_state()
            mqtt.publish(pub_ems_topic, pub_msg)
        elif split_msg[1].split('=')[0]== 'operation_mode': # operation_mode msg
            mode = int(split_msg[1].split('=')[1])
            if mode == 6:
                pub_msg = response_operation_mode(recv_index, mode)
            else:
                pref = float(split_msg[2].split('=')[1])
                pub_msg = response_operation_mode(recv_index, mode, pref)
            mqtt.publish(pub_ais_topic, pub_msg)
    else:
        print("Unknow Msg")


# mqtt connection
mqtt = mqttc.Client()
mqtt.on_message = on_message
mqtt.on_connect = on_connect
mqtt.on_publish = on_publish
mqtt.on_subscribe = on_subscribe
mqtt.connect(host=url, port=mqtt_port)
mqtt.subscribe(sub_topic, 0)

mqtt.loop_start()

while True:
    try:
        msg = input("event input ex)[conv green/red] or [batt green/red] :")
        split_msg = msg.split(' ')
        color = split_msg[1]
    except:
        print('input error')

    if split_msg[0] =='conv':
        pub_msg = event_converter_fault(color)
        mqtt.publish(pub_ems_topic, pub_msg)
    elif split_msg[0]=='batt':
        pub_msg = event_battery_fault(color)
        mqtt.publish(pub_ems_topic, pub_msg)


