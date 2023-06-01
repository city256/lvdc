import datetime
import string

import pandas as pd
import numpy as np
import csv
import re
import matplotlib.pyplot as plt

dataset = pd.read_csv('load_train.csv', encoding='CP949')


print(dataset)
#print(dataset.values[0])
data = pd.DataFrame(columns=['date','load'])
index = 0
day= 10

# year, month, hour, load 데이터를 year, month, day, hour, load 데이터 포맷으로 변경
for year, month, hour, load1, load2 in dataset.values:
    nmonth = re.sub(r'[^0-9]', '', month)
    nhour = re.sub(r'[^0-9]', '',hour)
    nhour = int(nhour) - 1
    try:
        for day in range(1, 32):
            date_time = '{0}-{1}-{2} {3}'.format(year, nmonth.zfill(2), str(day).zfill(2), str(nhour).zfill(2))
            date_time_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d %H')
            temp = pd.DataFrame(data=[[date_time_obj, round(load2 / 31, 2)]], columns=['date', 'load'])
            data = pd.concat([data, temp], ignore_index=True)

        index += 1
    except ValueError as ve:
        #print('timedate Value Error', ve)
        ve



data = data.sort_values(by='date', ignore_index=True)
data.to_csv('2020_load.csv')


data.plot()
plt.title('test Graph')
plt.legend()

plt.plot(data)
plt.xlabel("date")
plt.ylabel("load")


#data = pd.DataFrame(columns=['date','load'])

print(data)


