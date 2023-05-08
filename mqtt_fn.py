import paho.mqtt.client as mqtt
import config as cfg
import ml
from datetime import datetime

now = datetime.now()
mqttc = mqtt.Client()
global msg_index
msg_index=0

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

    # set 명령어 처리
    if (msg_str[0:3]=="set"):
        global msg_index
        msg_str=msg_str[4:]
        split_msg = msg_str.split("&")
        recv_index = int(split_msg[0].split('=')[1])

        # msg index 처리
        if (msg_index < recv_index):
            msg_index = recv_index
            print(f'index = {recv_index}')

            mode_type = int(split_msg[1].split('=')[1])
            # set msg 예) set?index=2&type=1&limit=21.5
            # msg type 별 처리
            if (mode_type == 1): # 최적모드
                pf = ml.optimize_mode()
                pub_msg = f'set?ems_index={msg_index}&type=5&power_reference={pf}'
                mqttc.publish(cfg.pub_topic, pub_msg)
            elif (mode_type == 2): # 피크제어
                limit = float(split_msg[2].split('=')[1])
                pf = ml.peak_mode(limit)
                pub_msg = f'set?ems_index={msg_index}&type=5&power_reference={pf}'
                mqttc.publish(cfg.pub_topic, pub_msg)
            elif (mode_type == 3): # 수요관리
                pf = ml.demand_mode()
                pub_msg = f'set?ems_index={msg_index}&type=5&power_reference={pf}'
                mqttc.publish(cfg.pub_topic, pub_msg)
            elif (mode_type == 4): # 태양광연계
                pf = ml.pv_mode()
                pub_msg = f'set?ems_index={msg_index}&type=5&power_reference={pf}'
                mqttc.publish(cfg.pub_topic, pub_msg)
            elif (mode_type == 5): # 수동제어
                pf = float(split_msg[2].split('=')[1])
                pub_msg = f'set?ems_index={msg_index}&type=5&power_reference={pf}'
                mqttc.publish(cfg.pub_topic, pub_msg)
                print(f'passive, power_reference={pf}')
            else:
                print("Unknown type msg")
        else:
            print('Ignore Index Msg')
    else:
        print('Unknown Msg')
