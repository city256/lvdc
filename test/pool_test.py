import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
data = pd.read_csv('../pqms_data.csv', parse_dates=['date'])

# datetime 컬럼을 인덱스로 설정
data.set_index('date', inplace=True)

# 특정 기간 동안의 데이터만 필터링 (예: '2022-01-01'부터 '2022-12-31'까지)
filtered_data = data['2023-08-26':'2023-09-25']

# 데이터를 시간별로 그룹화하고 평균값을 계산
hourly_data = filtered_data.resample('H').mean()

# 각 컬럼의 값을 가져옴
acdc = hourly_data['acdc']
load = hourly_data['load']
pv = hourly_data['pv']

# 그래프 그리기
plt.figure(figsize=(30,15))  # 그래프 크기를 수정

# 각 값에 대한 그래프를 그림
plt.plot(acdc.index, acdc, label='ACDC', marker='o', linestyle='-')
plt.plot(load.index, load, label='Load', marker='.', linestyle='--')
plt.plot(pv.index, pv, label='PV', marker='x', linestyle='-.')


# 그래프에 제목 및 라벨 추가
plt.title('PQMS Data')
plt.xlabel('Date', fontsize=30)
plt.ylabel('kWh',fontsize=30)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()  # 범례 표시
plt.tight_layout()
plt.show()