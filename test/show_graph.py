import pandas as pd
import matplotlib.pyplot as plt

def sum_data(data, start_time, end_time):
    filtered_data = data[start_time:end_time]
    print(filtered_data)

    # "acdc" 열의 합을 계산합니다.
    sum_acdc = filtered_data['acdc'].sum()
    sum_load = filtered_data['load'].sum()
    sum_pv = filtered_data['pv'].sum()
    sum_charge = filtered_data['ess_charge'].sum()
    sum_discharge = filtered_data['ess_discharge'].sum()
    print('ACDC = ', sum_acdc)
    print('Load = ', sum_load)
    print('PV = ', sum_pv)
    print('cha = ', sum_charge)
    print('dis = ', sum_discharge)
    return

# CSV 파일 읽기
data = pd.read_csv('../test/pqms_data_demand.csv', parse_dates=['date'])

# datetime 컬럼을 인덱스로 설정
data.set_index('date', inplace=True)

# 특정 기간 동안의 데이터만 필터링 (예: '2022-01-01'부터 '2022-12-31'까지)
filtered_data = data['2023-10-02 00:00:00':'2023-10-04 00:00:00']

# 데이터를 시간별로 그룹화하고 평균값을 계산
#filtered_data = filtered_data.resample('H').mean()
#sum_data(data, '2023-11-02 10:00:00', '2023-11-03 18:00:00')

# 각 컬럼의 값을 가져옴
acdc = filtered_data['acdc']
load = filtered_data['load']
pv = filtered_data['pv']
charge = filtered_data['ess_charge']
discharge = filtered_data['ess_discharge']

# 그래프 그리기
plt.figure(figsize=(30,15))  # 그래프 크기를 수정

# 각 값에 대한 그래프를 그림
plt.plot(acdc.index, acdc, 'm', label='Grid', marker='o', linestyle='-', linewidth='3')
plt.plot(load.index, load, 'darkorange', label='Load', marker='.', linestyle='-.', linewidth='2')
plt.plot(pv.index, pv, color='darkgreen', label='PV', marker='x', linestyle='-.', linewidth='2')
plt.plot(charge.index, charge, 'blue', label='Charge',  marker='v', linestyle='-', linewidth='3')
plt.plot(discharge.index, discharge, 'red', label='Discharge',  marker='s', linestyle='-', linewidth='3')


# 그래프에 제목 및 라벨 추가
plt.title('PQMS Data')
plt.xlabel('Date', fontsize=30)
plt.ylabel('kWh',fontsize=30)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=25, ncol=5, loc='upper left', frameon=True, shadow=True)  # 범례 표시
plt.tight_layout()
plt.show()