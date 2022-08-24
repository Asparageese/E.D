import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import scale
from sklearn.utils import shuffle

# single feature dataset generator

def classifier(current,future):
    if future > current:
        return 1
    else:
        return 0

# seperate the input sequences into negative and positive motions and mask.
# feed the separated masked input into two different encoder blocks.
def principal_decomposition(array):
    positive=[]
    negative=[]
    for i in tqdm(array,desc='princial_seperation'):
        if i > 0:
            positive.append(i)
            negative.append(0.)
        else:
            negative.append(i)
            positive.append(0.)
    return np.array(positive),np.array(negative)

def window_creation(array):
    windows = []
    for i in tqdm(range(np.size(array)),desc='time_series_windows'):
        if (i+prediction_entropy+input_size) == np.size(array):
            break
        windows.append(data[i:i+input_size])
    return np.array(windows)


input_size = 20
prediction_entropy = 4

data = pd.read_csv("eurusd_minute.csv")
data = np.diff(np.diff(data['BO'].to_numpy()))
data = scale(data,with_std=True)

labels = []
for i in range(np.size(data)):
    if (i+prediction_entropy+input_size) == np.size(data):
        break
    labels.append(classifier(data[i+input_size],data[i+input_size+prediction_entropy]))

p_set,n_set = principal_decomposition(data)
p_set,n_set = window_creation(p_set),window_creation(n_set)

p_set,n_set,labels = shuffle(p_set,n_set,labels,random_state=0)

np.save("Attention_playground/p_set.npy", p_set)
np.save("Attention_playground/n_set.npy", n_set)
np.save("Attention_playground/labels.npy", labels)






