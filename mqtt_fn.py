import paho.mqtt.client as mqtt
import operation_mode as op
mqttc = mqtt.Client()

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
    print("mid: " + str(mid))


def on_subscribe(client, obj, mid, granted_qos):
    print("Subscribe complete : " + str(mid) + " " + str(granted_qos))

# mqtt subscribe 메시지 처리
def msg_handler(msg):
    msg_str = str(msg)[2:-1] # payload 문자열처리

    # set 명령어 처리
    if (msg_str[0:3]=="set"):
        msg_str=msg_str[4:]
        split_msg = msg_str.split("&")
        mode_type = int(split_msg[1].split('=')[1])
        # set msg 예) set?index=2&type=1&limit=21.5
        # msg index 처리
        msg_index = int(split_msg[0].split('=')[1])
        print(msg_index)

        # msg type 별 처리
        if (mode_type == 1):
            op.optimize_mode()
        elif(mode_type == 2):
            limit = float(split_msg[2].split('=')[1])
            op.peak_mode(limit)
        elif (mode_type == 3):
            op.demand_mode()
        elif (mode_type == 4):
            op.pv_mode()
        elif (mode_type == 5):
            pf = float(split_msg[2].split('=')[1])
            op.passive_mode(pf)
        else:
            print("Unknown type msg")
    else:
        print('Unknown Msg')
