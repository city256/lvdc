import pandas as pd


csv_path = 'data/load.csv'

df1 = pd.read_csv(csv_path, encoding ='cp949')
df1.head(100)
print(df1.head())

