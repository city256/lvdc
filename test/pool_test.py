import time
import pandas as pd
import config as cfg
import datetime
import mqtt_fn


cfg.soc_index['231'] = 341.13
predict_until = pd.to_datetime(cfg.now_hour) + datetime.timedelta(days=7)
# pub_msg = f'get?p_index={mqtt_fn.pms_index}&soc_report'
# mqtt_fn.mqttc.publish(cfg.pub_pms_topic, pub_msg)
print(predict_until)

print(cfg.soc_index['231'])


#soc = 14.6
#soc = cfg.soc_index[mqtt_fn.pms_index]