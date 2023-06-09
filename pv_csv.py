import datetime
import string

import pandas as pd
import numpy as np
import csv
import re
import matplotlib.pyplot as plt

dataset = pd.read_csv('pv_data.csv', encoding='CP949')


print(dataset)
#print(dataset.values[0])
data = pd.DataFrame(columns=['date','pv'])
index = 0
day= 10
hour =[0]*24
print(hour)
for date, hour[0],hour[1],hour[2],hour[3],hour[4],hour[5],hour[6],hour[7],hour[8],hour[9],hour[10],hour[11],hour[12],hour[13],hour[14],hour[15],hour[16],hour[17],hour[18],hour[19],hour[20],hour[21],hour[22],hour[23] in dataset.values:
    #print(data, hour[14])
    for hi in range(24):
        if(hi!=24):
            date_time = '{} {}'.format(date, str(hi))
            date_time_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d %H')
            temp = pd.DataFrame(data=[[date_time_obj, hour[hi]]], columns=['date', 'pv'])
            data = pd.concat([data, temp], ignore_index=True)
        else:
            date_time = '{} 0'.format(date)
            date_time_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d %H')
            temp = pd.DataFrame(data=[[date_time_obj, hour[hi]]], columns=['date', 'pv'])
            data = pd.concat([data, temp], ignore_index=True)


data = data.sort_values(by='date', ignore_index=True)
data.to_csv('2020_pv.csv')

data.plot()
plt.title('test Graph')
plt.legend()

plt.plot(data)
plt.xlabel("date")
plt.ylabel("pv")

#data = pd.DataFrame(columns=['date','load'])

print(data)



