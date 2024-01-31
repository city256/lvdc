import time
import mqtt_fn
import config as cfg
import schedule
import data_collecter as dc

if __name__ == '__main__':
    # mqtt connection
    mqtt_fn.mqttc.on_message = mqtt_fn.on_message
    mqtt_fn.mqttc.on_connect = mqtt_fn.on_connect
    mqtt_fn.mqttc.on_publish = mqtt_fn.on_publish
    mqtt_fn.mqttc.on_subscribe = mqtt_fn.on_subscribe
    mqtt_fn.mqttc.connect(host=cfg.broker_url, port=cfg.mqtt_port)
    mqtt_fn.mqttc.subscribe(cfg.sub_topic, 0)
    mqtt_fn.mqttc.loop_start()

    # 주기적으로 예측 데이터 업데이트
    #schedule.every().hour.at("00:20").do(dc.update_csv)
    #schedule.every(10).minutes.at(":30").do(dc.update_csv)

    # init : 프로그램 시작시 예측 돌리는 변수 (0: 시작시 예측 / 1: 시작시 예측 안함)
    init = 1
    # 스케쥴링 루프문
    while True:
        if init==0:
            #dc.update_csv()
            init+=1
        #schedule.run_pending()
        time.sleep(1)

