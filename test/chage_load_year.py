import datetime
import string

import pandas as pd
import numpy as np
import csv
import re
import matplotlib.pyplot as plt

dataset = pd.read_csv('../load_2020.csv', encoding='CP949')

dataset['date']=pd.to_datetime(dataset['date'], format='%Y-%m-%d %H:%M:%S')
data = pd.DataFrame(columns=['date','load'])
#data = data.set_index('date')
date_format = "%Y-%m-%d %H:%M:%S"

print(data)
for x, date, load in dataset.values:

    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute

    try:
        date_str = datetime.datetime.strptime('{}-{}-{} {}:{}:00'.format(year+3, month, day, hour, minute), date_format)
        temp = pd.DataFrame(data=[[date_str, load*10]], columns=['date', 'load'])

        data = pd.concat([data, temp], ignore_index=True)
        #print(data)
    except ValueError as ve:
          None

    #date_2023 = date_obj +datetime.timedelta(days=365)

print(data)


# data = data.sort_values(by='date', ignore_index=True)
data.to_csv('../load_2023.csv')


#data = pd.DataFrame(columns=['date','load'])




