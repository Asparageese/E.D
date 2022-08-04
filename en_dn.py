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


data_window = 20
moment_entropy = 5
def time_series_generator(data):
    data = np.diff(data)
    label = []
    series_data = []
    for i in tqdm(range(np.size(data))):
        if i + moment_entropy + data_window == np.size(data):
            break
        label.append(classifier(data[i],data[i+moment_entropy+data_window])), series_data.append(data[i:i+data_window])
    return np.array(label), np.array(series_data)

labels, series_data = time_series_generator(featureA)
print(np.shape(series_data)), print(np.shape(labels))

del featureA
#################################################################

class ffn(tf.keras.layers.Layer):
    def __init__(self,output_dim):
        super(ffn,self).__init__()
        self.output_dim = output_dim
        self.de1 = tf.keras.layers.Dense(80, activation='relu')
        self.de2 = tf.keras.layers.Dense(units=output_dim, activation='relu')
    def call(self,x):
        x = self.de1(x)
        return self.de2(x)


class encoder_subroutine(tf.keras.layers.Layer):
    def __init__(self, output_dim,num_heads,key_dim,rate):
        super(encoder_subroutine, self).__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,attention_axes=(0,1))
        self.ffn = ffn(output_dim)

        self.drop = tf.keras.layers.Dropout(rate)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self,inputs):
        x = self.attention(inputs,inputs)
        x = self.drop(x)
        x = self.norm(x+inputs)
        x = self.ffn(x)
        x = self.drop(x)
        x = self.norm(x+inputs)
        x = self.attention(x,x)
        x = self.drop(x)
        return self.norm(x+inputs)
    

class decoder_subroutine(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_heads,key_dim,rate):
        super(decoder_subroutine, self).__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.rate = rate

        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,attention_axes=(0,1))
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,attention_axes=(0,1))

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


        self.ff1 = ffn(output_dim)

        self.drop1 = tf.keras.layers.Dropout(rate)


    def call(self,inputs):
        x = self.mha1(inputs,inputs)
        x = self.drop1(x)
        x = self.norm1(x+inputs)
        x = self.ff1(x)
        x = self.drop1(x)
        x = self.norm1(x+inputs)
        x = self.mha2(x,x)
        x = self.norm1(x+inputs)
        x = self.mha1(x,x)
        x = self.drop1(x)
        return self.norm1(x+inputs)


num_layers = 2
input_layer = tf.keras.layers.Input(shape=(data_window,))
enc_layers = [encoder_subroutine(output_dim=data_window,num_heads=moment_entropy,key_dim=data_window,rate=0.1) for i in range(num_layers)]
dec_layers = [decoder_subroutine(output_dim=data_window,num_heads=moment_entropy,key_dim=data_window,rate=0.1) for i in range(num_layers)]
x = input_layer
for i in range(num_layers):
    x = enc_layers[i](x)
for i in range(num_layers):
    x = dec_layers[i](x)
x = ffn(data_window)(x)

prediction = tf.keras.layers.Dense(units=1, activation='softmax')(x)
model = tf.keras.models.Model(inputs=input_layer, outputs=prediction, name='model')



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy','mse'])
model.summary()

with tf.device('/gpu:0'):
    model.fit(series_data,labels, epochs=1, validation_split=0.4, shuffle=True, batch_size=32)
