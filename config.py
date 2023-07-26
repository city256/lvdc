import datetime

# mqtt config
broker_url = 'lvdc.iptime.org'
sub_topic = 'lvdc/ais'
pub_pms_topic = 'lvdc/pms'
pub_ems_topic = 'lvdc/ems/pms'
mqtt_port = 1883

# maria db config
db_url = 'lvdc.iptime.org'
db_user = "root"
db_pw = "Lvdc12341@"
db_name = "etri_lvdc"
db_port = 13306

# time
now = datetime.datetime.now()
now_date = datetime.datetime.now().strftime('%Y-%m-%d')
now_hour = datetime.datetime.now().strftime('%Y-%m-%d %H')

# microgrid spec
ess_capacity = 1000
conv_capacity_1h = 250
pv_capacity = 250
conv_capacity_15m = 62.5
soc_min = 10
soc_max = 90

max_capacity = ess_capacity * (soc_max * 0.01)
min_capacity = ess_capacity * (soc_min * 0.01)

#soc, target_soc = 0
soc = 0.4
target_soc = 0.473


# ess 충방전, 목표 soc에 따른 power_reference 계산 (15분 단위)
ess_energy = ess_capacity * (target_soc - soc) # 양수는 충전필요량, 음수는 방전필요량
ess_energy = round(ess_energy, 2)
ess_time = ess_energy / (conv_capacity_1h * 0.25) # 15분 주기로 충방전량 나누기

'''
print(ess_time, ess_energy)
if(ess_time > 0): #양수, 충전
    if(ess_time > 1):
        p_ref = conv_capacity_15m
    else:
        p_ref = conv_capacity_15m * ess_time
elif(ess_time < 0): #음수, 방전
    if(ess_time < -1):
        p_ref = -conv_capacity_15m
    else:
        p_ref = conv_capacity_15m * ess_time
else: # 충방전 안함
    p_ref = 0

print(p_ref)
'''