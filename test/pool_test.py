import pandas as pd
import datetime

pv = pd.read_csv('../pred_pv.csv')
load = pd.read_csv('../pred_load.csv')

print(datetime.datetime.now().strftime('%Y-%m-%d %H:00:00'))
datetime.datetime.now()
print(load['load'][1])