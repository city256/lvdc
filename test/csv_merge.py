import pandas as pd

# CSV 파일 로드
quarter_hourly_df = pd.read_csv('../csv/pred_load.csv')  # 15분 주기 데이터
hourly_df = pd.read_csv('../csv/pred_pv.csv')                  # 1시간 주기 데이터

# datetime 열을 datetime 객체로 변환
quarter_hourly_df['date'] = pd.to_datetime(quarter_hourly_df['date'])
hourly_df['date'] = pd.to_datetime(hourly_df['date'])

# 1시간 데이터를 15분 간격으로 확장
rows = []
for index, row in hourly_df.iterrows():
    for i in range(4):
        new_row = row.copy()
        new_row['date'] = row['date'] + pd.Timedelta(minutes=15 * i)
        new_row['pv'] = row['pv'] / 4  # PV 값을 1/4로 나눔
        rows.append(new_row)

expanded_hourly_df = pd.DataFrame(rows)

# 'datetime'을 기준으로 병합하고, 'pv_hourly' 열만 추가
merged_df = pd.merge(quarter_hourly_df, expanded_hourly_df, on='date', how='left')

# 결과를 새로운 CSV 파일로 저장
merged_df.to_csv('merged_data.csv', index=False)