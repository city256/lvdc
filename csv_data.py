import string

import pandas as pd
import numpy as np
import csv
import re



dataset = pd.read_csv('load_train.csv', encoding='CP949')


print(dataset)
print(dataset.values[0])
data = pd.DataFrame(columns=['date','load'])
index = 0

for year, day, hour, load1, load2 in dataset.values:
    nday = re.sub(r'[^0-9]', '',day)
    nhour = re.sub(r'[^0-9]', '',hour)
    data.loc[index] = [str(year)+nday.zfill(2)+nhour.zfill(2),load2]
    index +=1
#data = pd.DataFrame(columns=['date','load'])

print(data)


