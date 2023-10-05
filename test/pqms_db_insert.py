import time
import paho.mqtt.client as mqtt
import config as cfg
import json
import db_fn as db
from datetime import datetime

mqttc = mqtt.Client()
sub_topic = 'lvdc/ems/pqms'
pub_topic = 'lvdc/pqms'

until_date = datetime.strptime('2023-09-26T14:00:00', '%Y-%m-%dT%H:%M:%S')

def on_connect(client, userdata, flags, rc):
    # print("rc: " + str(rc))
    if rc==0:
        print("connected OK Returned code=",rc)
    else:
        print("Bad connection Returned code=",rc)

# 브로커에게 메시지가 도착하면 on_message 실행 (이벤트가 발생하면 호출)
def on_message(client, obj, msg):
    #print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
    msg_handler(msg.payload)

def on_publish(client, obj, mid):
    # 용도 : publish를 보내고 난 후 처리를 하고 싶을 때
    # 사실 이 콜백함수는 잘 쓰진 않는다.
    print("Publish Complete: " + str(mid))

def on_subscribe(client, obj, mid, granted_qos):
    print("Subscribe Complete : " + str(mid) + " " + str(granted_qos))

def msg_handler(msg):
    msg_str = str(msg)[2:-1] # payload 문자열처리
    msg_str = msg_str.replace("\\r", "") # 개행문자 처리
    msg_str = msg_str.replace("\\n", "")
    msg_str = msg_str.replace("\\t", "")

    try:
        msg_json = json.loads(msg_str, strict=False) # payload json 처리
        date = datetime.strptime(msg_json['p_time'], '%Y-%m-%dT%H:%M:%S')
        print("json msg received")
    except json.JSONDecodeError as e:
        print(f"msg not json format: {e}")

    # pms get response 처리
    if (msg_json['p_type'] == 'get' and date < until_date and int(msg_json['p_contents']['index'])%15==0):
        db.put_pqms_data(msg_json)
        #print(f'get Response Msg= {msg_json}')
    else:
        #print(f'Unknown Msg= {msg_str}')
        print('')


# mqtt connection
mqttc.on_message = on_message
mqttc.on_connect = on_connect
mqttc.on_publish = on_publish
mqttc.on_subscribe = on_subscribe
mqttc.connect(host=cfg.broker_url, port=cfg.mqtt_port)
mqttc.subscribe(sub_topic, 0)
mqttc.loop_start()

#db.put_pqms_data_index('2023-09-26 15:00:00',900, 65627, 85186.8, 314.69, 536.21, 25362.7)

'''for j in range(7, 26):
    print(str(j).zfill(2))
    for i in range(96):
        print(i * 15)
        pub_msg = f'get?p_index=222&pqms_load&date=2023-09-{str(j).zfill(2)}&time_index={i * 15}'
        mqttc.publish('lvdc/pqms', pub_msg)
        time.sleep(0.5)'''
'''
for i in range(96):
    print(i * 15)
    pub_msg = f'get?p_index=222&pqms_load&date=2023-09-01&time_index={i * 15}'
    mqttc.publish('lvdc/pqms', pub_msg)
    time.sleep(0.5)

mqttc.loop_stop()
'''