import time
import pandas as pd
import config as cfg
import datetime
import json
import mqtt_fn


cfg.soc_index['231'] = 341.13
predict_until = pd.to_datetime(cfg.now_hour) + datetime.timedelta(days=7)
# pub_msg = f'get?p_index={mqtt_fn.pms_index}&soc_report'
# mqtt_fn.mqttc.publish(cfg.pub_pms_topic, pub_msg)
print(predict_until)

print(cfg.soc_index['231'])

json_str = '{\r\n\t"p_type": "set",\r\n\t"p_id": "MG1_SVR_PMS",\r\n\t"p_cmd": "response/operation_mode",\r\n\t"p_index": 2,\r\n\t"p_error": 0,\r\n\t"p_time": "2023-08-17T09:37:51",\r\n\t"p_contents": {\r\n\t\t"operation_mode": 1,\r\n\t\t"power_reference": -250\r\n\t}\r\n}'
json_str = json_str.replace("\'", "\"")
import re
p = re.compile('(?<!\\\\)\'')
json_str = p.sub('\"', json_str)
print(json_str)

my_dict = json.loads(json_str)
print(my_dict)  # 👉 <class 'dict'>

my_json_str = json.dumps(my_dict)
print(my_json_str)  # 👉 <class 'str'>
#soc = 14.6
#soc = cfg.soc_index[mqtt_fn.pms_index]