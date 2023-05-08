import ml
import config as cfg
import mqtt_fn

def optimize_mode():
    print("optimize_mode")
    return

def peak_mode(limit):
    print(f'peak_mode, limit= {limit}')
    index = 1
    pf = ml.calulate_pf(limit)
    pub_msg = f'set?ems_index={index}&type=5&power_reference={pf}'
    mqtt_fn.mqttc.publish(cfg.pub_topic, pub_msg)
    return

def demand_mode():
    print("demand_mode")
    return

def pv_mode():
    print("pv_mode")
    return

def passive_mode(pf):
    print(f'passive_mode, pf= {pf}')
    return

