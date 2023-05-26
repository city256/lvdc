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



for year, month, hour, load1, load2 in dataset.values:
    nmonth = re.sub(r'[^0-9]', '', month)
    nhour = re.sub(r'[^0-9]', '',hour)
    #data.loc[index] = [str(year) + nmonth.zfill(2) + nhour.zfill(2), load2]
    for day in range(1, 32):
        temp = pd.DataFrame(data=[
            ['{0}-{1}-{2} {3}'.format(year, nmonth.zfill(2), str(day).zfill(2), nhour.zfill(2)), round(load2 / 31, 2)]],
                            columns=['date', 'load'])
        data = pd.concat([data, temp], ignore_index=True)
    index +=1

data = data.sort_values(by='date', ignore_index=True)
data.to_csv('test.csv')


data.plot()
plt.title('test Graph')
#plt.plot(data['date'], data['load'], 'rs--')
plt.xlabel("date")
plt.ylabel("load")
plt.show()

#data = pd.DataFrame(columns=['date','load'])

print(data)


