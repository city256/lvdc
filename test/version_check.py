
import tensorflow as tf
print('tensor = ',tf.__version__)

import numpy
print('numpy = ',numpy.__version__)

import keras
print('keras = ',keras.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import pandas as pd
print('pandas = ',pd.__version__)