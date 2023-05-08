import mqtt_fn
import config as cfg

# mqtt connection
mqtt_fn.mqttc.on_message = mqtt_fn.on_message
mqtt_fn.mqttc.on_connect = mqtt_fn.on_connect
mqtt_fn.mqttc.on_publish = mqtt_fn.on_publish
mqtt_fn.mqttc.on_subscribe = mqtt_fn.on_subscribe
mqtt_fn.mqttc.connect(host=cfg.url, port=cfg.mqtt_port)
mqtt_fn.mqttc.subscribe(cfg.sub_topic, 0)

print('main page')
mqtt_fn.mqttc.loop_forever()

