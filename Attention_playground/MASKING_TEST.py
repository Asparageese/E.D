from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Masking, GRU, Bidirectional, Dense, Flatten, Dropout, BatchNormalization, Concatenate

class dense_consideration(tf.keras.layers.Layer):
    def __init__(self,out_dim,rate):
        super(dense_consideration, self).__init__()
        self.out_dim = out_dim
        self.rate = rate

        self.d1 = Dense(self.out_dim,activation='relu')
        self.drop = Dropout(self.rate)
        self.norm = BatchNormalization()

        self.d2 = Dense(self.out_dim, activation='relu')
        self.norm = BatchNormalization()

        self.d3 = Dense(self.out_dim, activation='relu')
        self.norm = BatchNormalization()
    def __call__(self, inputs):
        x = self.d1(inputs)
        x = self.drop(x)
        x = self.norm(x)
        x = self.d2(x)
        x = self.drop(x)
        x = self.norm(x)
        x = self.d3(x)
        x = self.drop(x)
        x = self.norm(x)
        return x
class GRU_consideration(tf.keras.layers.Layer):
    def __init__(self,out_dim,rate):
        super(GRU_consideration, self).__init__()
        self.out_dim = out_dim
        self.rate = rate

        self.gru1 = GRU(self.out_dim,return_sequences=True,return_state=True, activation='tanh')
        self.gru2 = GRU(self.out_dim,return_sequences=True,return_state=True, activation='tanh')
        self.gru3 = GRU(self.out_dim, activation='tanh')

    def __call__(self, inputs):
        x = self.gru1(inputs)
        x = self.gru2(x)
        x = self.gru3(x)
        return x


batch_size = 32
epochs = 3
input_size = 20
prediction_entropy = 4

### import data
p_data = np.load('p_set.npy')
n_data = np.load('n_set.npy')
labels = np.load('labels.npy')
p_data, n_data = p_data[:,:,tf.newaxis], n_data[:,:,tf.newaxis]

p_data,n_data, labels = shuffle(p_data,n_data,labels,random_state=0)

input_layerA = Input(shape=(p_data.shape[1:]),name='in_p')
input_layerB = Input(shape=(n_data.shape[1:]),name='in_n')
maskA = Masking(mask_value=0.)(input_layerA)
maskB = Masking(mask_value=0.)(input_layerB)

dp = GRU_consideration(64,0.2)(maskA)
dn = GRU_consideration(64,0.2)(maskB)

conc = Concatenate()([dp,dn])

yhat = Dense(1,activation='softmax')(conc)
model = tf.keras.Model(inputs=[input_layerA,input_layerB],outputs=[yhat],name='test_model')
model.summary()

with tf.device('CPU:0'):
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(),optimizer=tf.keras.optimizers.Nadam(),metrics=['accuracy'])
with tf.device('GPU:0'):
    model.fit({"in_p":p_data,"in_n":n_data},labels,batch_size=batch_size,epochs=epochs,shuffle=True,verbose=1,validation_split=0.3)