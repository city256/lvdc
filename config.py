# mqtt config
url = 'lvdc.iptime.org'
sub_topic = 'lvdc/ais'
pub_pms_topic = 'lvdc/ess'
pub_ems_topic = 'lvdc/ems/ess'
mqtt_port = 1883

# maria db config
db_url = 'lvdc.iptime.org'
db_user = "root"
db_pw = "Lvdc12341@"
db_name = "etri_lvdc"
db_port = 13306

# microgrid spec
ess_capacity = 800
conv_capacity_1h = 250
conv_capacity_15m = 62.5
pv_capacity = 250

#soc, target_soc = 0
soc = 0.4
target_soc = 0.447
# 15분 주기로 target_soc 결정하는 알고리즘 필요
# 예측 태양광발전량 - 예측 전력사용량 = 양수충전/음수방전
# 예) 500 - 400 = 100kWh 방전 (100/4)
# 15분 동안 방전해야할 양 25kW
# 이때 충/방전량이 62.5kW가 넘어가면 15분 최대 출력으로

# ess 충방전, 목표 soc에 따른 power_reference 계산 (15분 단위)
ess_energy = ess_capacity * (target_soc - soc) # 양수는 충전필요량, 음수는 방전필요량
ess_energy = round(ess_energy, 2)
ess_time = ess_energy / conv_capacity_15m # 15분 주기로 충방전량 나누기

print(f'충방전 에너지량(kW): {ess_energy}, 충방전 시간(15분 단위): {ess_time}')
if(ess_time > 0): #양수, 충전
    if(ess_energy >= conv_capacity_15m):
        p_ref = conv_capacity_15m
    else:
        p_ref = ess_energy
elif(ess_time < 0): #음수, 방전
    if(ess_energy <= -conv_capacity_15m):
        p_ref = -conv_capacity_15m
    else:
        p_ref = ess_energy
else: # 충방전 안함
    p_ref = 0

print(p_ref)

if (ess_time > 1):
    p_ref = conv_capacity_15m
else:
    p_ref = conv_capacity_15m * ess_time
if(ess_time < -1):
    p_ref = -conv_capacity_15m
else:
    p_ref = conv_capacity_15m * ess_time

# pcs kW module kW가 별개로 있음
