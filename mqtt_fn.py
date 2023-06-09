import paho.mqtt.client as mqtt
import config as cfg
import ml
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
        json_str = json.loads(msg_str)
        print("json msg received")
    except json.JSONDecodeError as e:
        print(f"msg not json format: {e}")

    # set 명령어 처리
    if (msg_str[0:3]=="set"):
        global p_index
        msg_str=msg_str[4:]
        split_msg = msg_str.split("&")
        recv_index = int(split_msg[0].split('=')[1]) # p_index value
        mode_type = int(split_msg[1].split('=')[1]) # type value

        # p_index 처리
        if (split_msg[0].split('=')[0] == 'p_index' and p_index < recv_index):
            p_index = recv_index

            print(f'p_index = {recv_index}, type = {mode_type}')
            # set msg 예) set?p_index=2&type=1&limit=21.5
            # msg type 별 처리
            if (split_msg[1].split('=')[0]=='type' and mode_type == 1): # 최적모드
                pf = ml.optimize_mode()
                pub_msg = f'set?p_index={p_index}&type={mode_type}&power_reference={pf}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='type' and mode_type == 2): # 피크제어
                limit = float(split_msg[2].split('=')[1])
                pf = ml.peak_mode(limit)
                pub_msg = f'set?p_index={p_index}&type={mode_type}&power_reference={pf}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='type' and mode_type == 3): # 수요관리
                pf = ml.demand_mode()
                pub_msg = f'set?p_index={p_index}&type={mode_type}&power_reference={pf}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='type' and mode_type == 4): # 태양광연계
                pf = ml.pv_mode()
                pub_msg = f'set?p_index={p_index}&type={mode_type}&power_reference={pf}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
            elif (split_msg[1].split('=')[0]=='type' and mode_type == 5): # 수동제어
                pf = float(split_msg[2].split('=')[1])
                pub_msg = f'set?p_index={p_index}&type={mode_type}&power_reference={pf}'
                mqttc.publish(cfg.pub_pms_topic, pub_msg)
                print(f'passive, power_reference={pf}')
            else:
                print("Unknown type msg")
        else:
            print('Ignore Index Msg')

    # pms set response 처리
    elif (json_str['p_type'] == 'set' and json_str['p_cmd'] == 'response'):
        print(f'set Response Msg= {json_str}')
        mqttc.publish(cfg.pub_ems_topic, str(json_str))

    # pms get response 처리
    elif (json_str['p_type'] == 'get' and json_str['p_cmd'] == 'response'):
        print(f'get Response Msg= {json_str}')

    else:
        print(f'Unknown Msg= {msg_str}')
