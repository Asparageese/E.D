import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import scale
from sklearn.utils import shuffle

def classify(x,y):
    if x < y:
        return 0
    else:
        return 1

def window_formation(array):
    window = []
    for f in range(np.size(array[0, :])):
        for i in tqdm(range(np.size(array[:, f]))):
            if (i + input_size + prediction_size) == np.size(array[:, 0]):
                break
            window.append(array[i:i + input_size, f])
    return window

input_size = 12
prediction_size = 6

data = pd.read_csv('eurusd_minute.csv')
data = data[['BO','BC','AO','AC']].to_numpy()
print(np.shape(data))

p = []
n = []
for f in range(np.size(data[0,:])):
    p_set = []
    n_set = []
    processed_features = scale(np.diff(data[:,f]),with_std=True)
    for i in tqdm(range(np.size(processed_features))):
        if processed_features[i] > 0. :
            p_set.append(processed_features[i])
            n_set.append(0.)
        else:
            p_set.append(0.)
            n_set.append(processed_features[i])
    p.append(p_set), n.append(n_set)
p, n = np.swapaxes(p,0,1), np.swapaxes(n,0,1)
p, n = np.reshape(window_formation(p),(-1,input_size,4)), np.reshape(window_formation(n),(-1,input_size,4))

labels = []
processed_features = scale(np.diff(data[:,1]),with_std=True)
for i in range(np.size(processed_features)):
    if (i + input_size + prediction_size) == np.size(processed_features):
        break
    labels.append(classify(processed_features[i+input_size],processed_features[i+input_size+prediction_size]))
labels = np.array(labels)

print(np.shape(labels))
print(np.shape(p))

p,n,labels = shuffle(p,n,labels,random_state=0)

np.save('Attention_playground/p_multi',p)
np.save('Attention_playground/n_multi',n)
np.save('Attention_playground/labels',labels)


