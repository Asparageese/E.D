import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque


input_size = 40
prediction_entropy = 10

data = pd.read_csv("eurusd_minute.csv")
data = data.drop(columns=['ACh','BCh','Date','Time'])

def classify(future,current):
    if future > current:
        return 1
    if future == current:
        return 0
    else:
        return -1

    
data = data.to_numpy()
label = deque()
seq_data = deque()
for f in range(np.size(data[0,:1])):
    feature = np.array(data[:,f])
    seq_data.append(feature)
    for i in tqdm(range(np.size(data[:,0]))):
        if i + input_size + prediction_entropy == np.size(data[:,0]):
            break
        future = data[i+input_size+prediction_entropy,f]
        current = data[i,f]
        score = classify(future,current)
        
        
        label.append(score)

del data
seq_data = np.array(seq_data)        
#### indexing routing 

index_matrix = []
for i in tqdm(range(np.size(seq_data[0,:]))):	
    if i + input_size  == np.size(seq_data[0,:]):
        break
    index = np.arange(i,i+input_size)
    index_matrix.append(index)



label = np.array(label)
x_data = deque()
y_data = deque()
for i in range(np.size(index_matrix)):
    if i + input_size == np.size(index_matrix):
        break
    seriesA = np.take(seq_data,indices=index_matrix[i])
    x_data.append(seriesA)
del seq_data

for i in range(np.size(index_matrix)):
    if i + input_size == np.size(index_matrix):
        break
    seriesA = np.take(label,indices=index_matrix[i])
    y_data.append(seriesA)


x_data = np.array(x_data)
y_data = np.array(y_data)


print(np.shape(x_data),np.shape(y_data))