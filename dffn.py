import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm 
from sklearn.utils import shuffle

data = pd.read_csv('eurusd_minute.csv')
print("data_loaded")
data = data.drop(columns=['Date','Time','ACh','BCh'])


# single feature selection

featureA  = data['BO'].to_numpy()
print(np.shape(featureA))

def classifier(x,y):
    if y > x:
        return 1
    else:
        return 0

# label creation from the classifier function


data_window = 10
moment_entropy = 5
def time_series_generator(data):
    labels = []
    series_data = []
    for i in tqdm(range(np.size(data))):
        if i + data_window + moment_entropy == np.size(data):
            break
        labels.append(classifier(data[i],data[i+data_window+moment_entropy]))
        series_data.append(data[i:i+data_window])

    return np.array(series_data), np.array(labels)
series_data, labels = time_series_generator(featureA)


print(series_data.shape, labels.shape)


del featureA
#################################################################


class DFFN(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(DFFN, self).__init__()
        self.output_dim = output_dim

        self.de1 = tf.keras.layers.Dense(units=output_dim+25, activation='relu')
        self.d1 = tf.keras.layers.Dropout(0.2)
        self.n1 = tf.keras.layers.BatchNormalization()

        self.de2 = tf.keras.layers.Dense(units=output_dim+25, activation='relu')
        self.d2 = tf.keras.layers.Dropout(0.2)
        self.n2 = tf.keras.layers.BatchNormalization()

        self.de3 = tf.keras.layers.Dense(units=output_dim, activation='relu')

    def call(self,x):
        x = self.de1(x)
        x = self.d1(x)
        x = self.n1(x)
        x = self.de2(x)
        x = self.d2(x)
        x = self.n2(x)
        return self.de3(x)

input_layer = tf.keras.layers.Input(shape=(data_window,),dtype='float32')

feed_forward_network = DFFN(10)(input_layer)

prediction = tf.keras.layers.Dense(units=1, activation='softmax')(feed_forward_network)

model = tf.keras.models.Model(inputs=input_layer, outputs=prediction)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy','mse'])
model.summary()

with tf.device('/gpu:0'):
    model.fit(series_data,labels, epochs=1, validation_split=0.4, verbose=1, shuffle=True, batch_size=64)

