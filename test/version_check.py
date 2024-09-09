
import tensorflow as tf
print('tensor = ',tf.__version__)

import numpy
print('numpy = ',numpy.__version__)

import keras
print('keras = ',keras.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import pandas as pd
import matplotlib.pyplot as plt
def show_graph(csv_file):
    # CSV 파일 로드
    data = pd.read_csv(csv_file)

    # grid 계산 (load - pv)
    data['grid'] = data['load'] - data['pv']

    # date를 datetime 형식으로 변환
    data['date'] = pd.to_datetime(data['date'])

    # 그래프 그리기
    plt.figure(figsize=(14, 8))

    # Load, PV, Grid 값을 플롯
    plt.plot(data['date'], data['load'], label="Load", color="blue")
    plt.plot(data['date'], data['pv'], label="PV", color="orange")
    plt.plot(data['date'], data['grid'], label="Grid (load - pv)", color="green")

    # 그래프 레이블 및 타이틀 설정
    plt.xlabel("Date")
    plt.ylabel("Power (kW)")
    plt.title("Load, PV, and Grid Over Time")

    # 범례 및 그리드 추가
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 그래프 보여주기
    plt.show()

# 예시 사용
show_graph('../test/0907_0911.csv')