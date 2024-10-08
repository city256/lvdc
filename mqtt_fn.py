import time

import paho.mqtt.client as mqtt
import config as cfg
import operation_mode
import json

mqttc = mqtt.Client()
global p_index, pms_index
p_index=0
pms_index=0
test_scaling = 15
time_scaling = 4

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
    print("Publish Complete: " + str(mid))

def on_subscribe(client, obj, mid, granted_qos):
    print("Subscribe Complete : " + str(mid) + " " + str(granted_qos))

def get_soc_report():
    global pms_index
    pub_msg = f'get?p_index={pms_index}&soc_report'
    mqttc.publish(cfg.pub_pms_topic, pub_msg)
    pms_index += 1

# mqtt subscribe 메시지 처리
def msg_handler(msg):
    msg_str = str(msg)[2:-1] # payload 문자열처리
    msg_str = msg_str.replace("\\r", "") # 개행문자 처리
    msg_str = msg_str.replace("\\n", "")
    msg_str = msg_str.replace("\\t", "")

    # set 명령어 처리
    if (msg_str[0:3]=="set"):
        global p_index
        msg_str=msg_str[4:]
        split_msg = msg_str.split("&")
        recv_index = int(split_msg[0].split('=')[1]) # p_index value
        mode = int(split_msg[1].split('=')[1]) # type value
        # print('get_soc_report send')
        # p_index 처리
        if (split_msg[0].split('=')[0] == 'p_index' or p_index < recv_index):
            p_index = recv_index
            print(f'p_index = {recv_index}, operation_mode = {mode}')
            # msg type 별 처리
            if (split_msg[1].split('=')[0]=='operation_mode' and mode == 1): # 최적모드
                pf = operation_mode.optimize_mode()
                print('pf = ', pf * time_scaling / test_scaling)
                pub_msg = f'set?p_index={p_index}&operation_mode={mode}&power_reference={round(pf * time_scaling / test_scaling,2)}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='operation_mode' and mode == 2): # 피크제어
                limit = float(split_msg[2].split('=')[1])
                pf = operation_mode.peak_mode(limit)
                print('pf = ', pf * time_scaling / test_scaling)
                pub_msg = f'set?p_index={p_index}&operation_mode={mode}&power_reference={round(pf * time_scaling / test_scaling,2)}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='operation_mode' and mode == 3): # 수요관리
                pf = operation_mode.demand_mode()
                print('pf = ', pf)
                pub_msg = f'set?p_index={p_index}&operation_mode={mode}&power_reference={round(pf * time_scaling / test_scaling,2)}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='operation_mode' and mode == 4): # 태양광연계
                pf = operation_mode.pv_mode()
                print('pf = ', pf)
                pub_msg = f'set?p_index={p_index}&operation_mode={mode}&power_reference={round(pf * time_scaling / test_scaling,2)}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='operation_mode' and mode == 5): # 수동제어
                pf = float(split_msg[2].split('=')[1])
                pub_msg = f'set?p_index={p_index}&operation_mode={mode}&power_reference={pf}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
                print(f'passive, power_reference={pf}')
            elif (split_msg[1].split('=')[0]=='operation_mode' and mode == 6): # 독립모드
                pub_msg = f'set?p_index={p_index}&operation_mode={mode}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='operation_state'): # 동작 제어
                pub_msg = f'set?p_index={p_index}&operation_state={mode}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            else:
                print("Unknown operation_mode msg")
            pass
        else:
            print('Ignore Index Msg')
            pass

    try:
        msg_json = json.loads(msg_str, strict=False)  # payload json 처리
        print("json msg received")

        # pms set response 처리
        if msg_json['p_type'] == 'set':
            if msg_json['p_cmd'] == 'response/operation_mode':
                mqttc.publish(cfg.pub_ems_topic, msg_str)
            elif msg_json['p_cmd'] == 'response/operation_state':
                mqttc.publish(cfg.pub_ems_topic, msg_str)
            else:
                print(f'Unknown Response Msg= {msg_json}')
            pass

        # pms get response 처리
        elif (msg_json['p_type'] == 'get' and msg_json['p_cmd'] == 'response/soc_report'):
            print(cfg.soc_index)
            print(f'get Response Msg= {msg_json}')
            cfg.soc_index[msg_json['p_index']] = float(msg_json['p_contents']['current_soc'])
            pass

        else:
            print(f'Unknown Msg= {msg_str}')
            pass
    except json.JSONDecodeError as e:
        print(f"msg not json format: {e}")



