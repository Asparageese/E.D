import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque


input_size = 30
prediction_entropy = 8

data = pd.read_csv("eurusd_minute.csv")
data = data.drop(columns=['ACh','BCh','Date','Time'])

def classify(future,current):
    if future > current:
        return 1
    else:
        return 0

#handle single feature associations
def series_formation(array):
    series = []
    for i in tqdm(range(np.size(array))):
        if (i+input_size+prediction_entropy) == np.size(array):
            break
        series.append(array[i:i+input_size])
    return np.array(series)

def label_formation(array):
    labels = []
    for i in tqdm(range(np.size(array))):
        if (i+input_size+prediction_entropy) == np.size(array):
            break
        labels.append(classify(array[i+input_size+prediction_entropy],array[i+input_size]))
    return np.array(labels)

data = data['BO'].to_numpy()

f1 = series_formation(data)
labels = label_formation(data)

np.save('f1.npy',f1)
np.save('labels.npy',labels)




