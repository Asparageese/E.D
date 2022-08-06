import numpy as np 
from tqdm import tqdm 
import pandas as pd
import tensorflow as tf
import time

num_layers = 2
d_model = 20
moment_entropy = 3
key_dim = d_model + d_model
num_heads = num_layers


def classification(x,y):
    if x > y:
        return 0
    else:
        return 1




## form series and labels

data = pd.read_csv('eurusd_minute.csv')
data = data['BO'].to_numpy()
labels, series_data, V_PosE = [], [], []

data = np.diff(np.diff(data))
for i in tqdm(range(np.size(data))):
    if i + d_model + moment_entropy == np.size(data):
        break
    labels.append(classification(data[i],data[i+d_model+moment_entropy])), series_data.append(data[i:i+d_model]),


series_data,labels = np.array(series_data), np.array(labels)

#  a masking tensor will be created to mask input depending on the length of the input and number of layers.
#  The masking tensor should take form, [Obscure1,obscure2,...,FULL_SEQUENCE]. The idea is that by limiting some layers to partial information, we can force
#  recognition of temporal importance.


############### layer library ###############
class encoder_subroutine(tf.keras.layers.Layer):
    def __init__(self,latent_representation,density,num_heads,key_dim,rate):
        super(encoder_subroutine,self).__init__()
        self.latent_representation = latent_representation
        self.density = density
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.rate = rate

        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,key_dim=self.key_dim,attention_axes=(0,1))
        self.d1 = tf.keras.layers.Dense(self.density,activation='relu')
        self.d2 = tf.keras.layers.Dense(self.latent_representation,activation='relu')

        self.drop1 = tf.keras.layers.Dropout(self.rate)
        self.norm1 = tf.keras.layers.LayerNormalization()

    def call(self,inputs):
        x = self.mha1(inputs,inputs)
        x = self.drop1(x)
        xn = self.norm1(x+inputs)
        x = self.d1(x)
        x = self.d2(x)
        x = self.drop1(x)
        return self.norm1(x+xn)

class decoder_subroutine(tf.keras.layers.Layer):
    def __init__(self,d_model,density,num_heads,key_dim,rate):
        super(decoder_subroutine,self).__init__()
        self.d_model = d_model
        self.density = density
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.rate = rate

        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,key_dim=self.key_dim,attention_axes=(0,1))
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,key_dim=self.key_dim,attention_axes=(0,1))

        self.d1 = tf.keras.layers.Dense(self.density,activation='relu')
        self.d2 = tf.keras.layers.Dense(self.d_model,activation='relu')

        self.drop1 = tf.keras.layers.Dropout(self.rate)
        self.norm1 = tf.keras.layers.LayerNormalization()

    def call(self,enc_inputs,pos_data):
        y = self.mha1(pos_data,pos_data)
        y = self.drop1(y)
        yn = self.norm1(y+pos_data)
        x = self.mha2(enc_inputs,yn)
        #x = self.drop1(x)
        xn = self.norm1(x+yn)
        x = self.d1(xn)
        x = self.d2(x)
        x = self.drop1(x)
        return self.norm1(x+xn)


class transformer(tf.keras.models.Model):
    def __init__(self,d_model,density,num_heads,key_dim,rate):
        super(transformer,self).__init__()
        self.d_model = d_model
        self.density = density
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.rate = rate

        self.e_layers = [encoder_subroutine(self.d_model,self.density,self.num_heads,self.key_dim,self.rate) for i in range(num_layers)]
        self.d_layers = [decoder_subroutine(self.d_model,self.density,self.num_heads,self.key_dim,self.rate) for i in range(num_layers)]

        self.d1 = tf.keras.layers.Dense(self.density,activation='relu')
        self.d2 = tf.keras.layers.Dense(self.d_model,activation='relu')

        #self.drop1 = tf.keras.layers.Dropout(self.rate)
        self.norm1 = tf.keras.layers.LayerNormalization()
    def call(self,inputs):
        x = inputs
        for i in range(num_layers):
            xe = self.e_layers[i](x)
            xd = self.d_layers[i](xe,inputs)
            #x = self.drop1(xd)
            x = self.norm1(xd+inputs)
        return x



###############################################

input_layer = tf.keras.layers.Input(shape=(d_model,))
t_inference = transformer(d_model=d_model,density=128,num_heads=num_heads,key_dim=key_dim,rate=0.3)(input_layer)



lin_ff = tf.keras.layers.Dense(moment_entropy,activation='relu')(t_inference)
prediction = tf.keras.layers.Dense(1,activation='softmax')(lin_ff)

model = tf.keras.models.Model(inputs=input_layer,outputs=prediction)
model.summary()

with tf.device('/cpu:0'):
    
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy','mse'])
    model.fit(series_data,labels,epochs=2,batch_size=64,validation_split=0.2,verbose=1)


        

