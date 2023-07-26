import pandas as pd
import config as cfg
import datetime

pv = pd.read_csv('../pred_pv.csv')
load = pd.read_csv('../pred_load.csv')

date_str = (cfg.now + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')

predWL = float(load.loc[(load['date'] == date_str), 'load'])
predWPV = float(pv.loc[cfg.now.hour]['pv'])

print(predWL)
print(predWPV)