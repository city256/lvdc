import paho.mqtt.client as mqtt
import config as cfg
import operation_mode
from datetime import datetime
import json

now = datetime.now()
mqttc = mqtt.Client()
global p_index, pms_index
p_index=0
pms_index=0

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

# mqtt subscribe 메시지 처리
def msg_handler(msg):
    msg_str = str(msg)[2:-1] # payload 문자열처리
    try:
        msg_json = json.loads(msg_str) # payload json 처리
        print("json msg received")
    except json.JSONDecodeError as e:
        print(f"msg not json format: {e}")

    # set 명령어 처리
    if (msg_str[0:3]=="set"):
        global p_index
        msg_str=msg_str[4:]
        split_msg = msg_str.split("&")
        recv_index = int(split_msg[0].split('=')[1]) # p_index value
        mode = int(split_msg[1].split('=')[1]) # type value

        # p_index 처리
        if (split_msg[0].split('=')[0] == 'p_index' or p_index < recv_index):
            p_index = recv_index

            print(f'p_index = {recv_index}, operation_mode = {mode}')
            # set msg 예) set?p_index=2&type=1&limit=21.5
            # msg type 별 처리
            if (split_msg[1].split('=')[0]=='operation_mode' and mode == 1): # 최적모드
                pf = operation_mode.optimize_mode()
                print('pf = ', pf)
                pub_msg = f'set?p_index={p_index}&operation_mode={mode}&power_reference={pf}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='operation_mode' and mode == 2): # 피크제어
                limit = float(split_msg[2].split('=')[1])
                pf = operation_mode.peak_mode(limit)
                pub_msg = f'set?p_index={p_index}&operation_mode={mode}&power_reference={pf}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='operation_mode' and mode == 3): # 수요관리
                pf = operation_mode.demand_mode()
                pub_msg = f'set?p_index={p_index}&operation_mode={mode}&power_reference={pf}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='operation_mode' and mode == 4): # 태양광연계
                pf = operation_mode.pv_mode()
                pub_msg = f'set?p_index={p_index}&operation_mode={mode}&power_reference={pf}'
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
        else:
            print('Ignore Index Msg')

    # pms set response 처리
    elif msg_json['p_type'] == 'set':
        if msg_json['p_cmd'] == 'response/operation_mode':
            mqttc.publish(cfg.pub_ems_topic, msg_str)
        elif msg_json['p_cmd'] == 'response/operation_state':
            mqttc.publish(cfg.pub_ems_topic, msg_str)
        else :
            print(f'Unknown Response Msg= {msg_json}')

    # pms get response 처리
    elif (msg_json['p_type'] == 'get' and msg_json['p_cmd'] == 'response/soc_report'):
        print(f'get Response Msg= {msg_json}')
    else:
        print(f'Unknown Msg= {msg_str}')

