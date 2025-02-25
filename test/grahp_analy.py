import pandas as pd
import matplotlib.pyplot as plt


def sum_data_multiple_periods(data, periods):
    sum_results = []

    for start_time, end_time in periods:
        # 특정 기간 동안의 데이터 필터링
        filtered_data = data[start_time:end_time]

        # 각 열의 합계를 계산
        sum_acdc = filtered_data['acdc'].sum()
        sum_load = filtered_data['load'].sum()
        sum_pv = filtered_data['pv'].sum()
        sum_charge = filtered_data['ess_charge'].sum()
        sum_discharge = filtered_data['ess_discharge'].sum()

        # 결과를 리스트에 저장
        sum_results.append({
            'start_time': start_time,
            'end_time': end_time,
            'sum_acdc': sum_acdc,
            'sum_load': sum_load,
            'sum_pv': sum_pv,
            'sum_charge': sum_charge,
            'sum_discharge': sum_discharge
        })

        # 결과 출력
        print(f"\n기간: {start_time} ~ {end_time}")
        print('ACDC 합계 = ', sum_acdc)
        print('Load 합계 = ', sum_load)
        print('PV 합계 = ', sum_pv)
        print('Charge 합계 = ', sum_charge)
        print('Discharge 합계 = ', sum_discharge)

        # 그래프 그리기
        plot_data(filtered_data, start_time, end_time)

    return sum_results


def plot_data(filtered_data, start_time, end_time):
    # 각 컬럼의 값을 가져옴
    acdc = filtered_data['acdc']
    load = filtered_data['load']
    pv = filtered_data['pv']
    charge = filtered_data['ess_charge']
    discharge = filtered_data['ess_discharge']

    # 그래프 그리기
    plt.figure(figsize=(15, 7))  # 그래프 크기를 수정

    # 각 값에 대한 그래프를 그림
    plt.plot(acdc.index, acdc, 'm', label='Grid', marker='o', linestyle='-', linewidth='2')
    plt.plot(load.index, load, 'darkorange', label='Load', marker='.', linestyle='-.', linewidth='1.5')
    plt.plot(pv.index, pv, color='darkgreen', label='PV', marker='x', linestyle='-.', linewidth='1.5')
    plt.plot(charge.index, charge, 'blue', label='Charge', marker='v', linestyle='-', linewidth='2')
    plt.plot(discharge.index, discharge, 'red', label='Discharge', marker='s', linestyle='-', linewidth='2')

    # 그래프에 제목 및 라벨 추가
    plt.title(f'PQMS Data ({start_time} ~ {end_time})')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('kWh', fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10, ncol=3, loc='upper left', frameon=True, shadow=True)  # 범례 표시
    plt.tight_layout()
    plt.show()


# CSV 파일 읽기
data = pd.read_csv('../test/pqms_data_demand.csv', parse_dates=['date'])

# datetime 컬럼을 인덱스로 설정
data.set_index('date', inplace=True)

# 여러 기간에 대한 합계를 계산하고 저장
periods = [
    ('2023-10-02 00:00:00', '2023-10-04 12:00:00')
    # 추가 기간을 여기에 추가할 수 있습니다.
]

# 여러 기간에 대한 합계를 계산
sum_results = sum_data_multiple_periods(data, periods)
