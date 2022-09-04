from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Masking, Dense, Dropout, BatchNormalization, Concatenate

class simp_enc(tf.keras.layers.Layer):
    def __init__(self, out_dim, rate):
        super(simp_enc, self).__init__()
        self.out_dim = out_dim
        self.rate = rate

        self.d1 = Dense(self.out_dim,activation="relu")
        self.drop = Dropout(self.rate)
        self.norm = BatchNormalization()
        self.d2 = Dense(self.out_dim,activation="relu")
        self.d3 = Dense(self.out_dim,activation="relu")
    def call(self,inputs):
        x = self.d1(inputs)
        x = self.drop(x)
        x = self.norm(x)
        x = self.d2(x)
        x = self.drop(x)
        x = self.norm(x)
        return self.d3(x)





batch_size = 64
epochs = 3
prediction_entropy = 2

### import data
p_data = np.load('p_multi.npy')
n_data = np.load('n_multi.npy')
labels = np.load('labels.npy')
input_size = 30
#p_data, n_data = p_data[:,:,tf.newaxis], n_data[:,:,tf.newaxis]

p_data,n_data, labels = shuffle(p_data,n_data,labels,random_state=0)

input_layerA = Input(shape=(p_data.shape[1:]),name='in_p')
input_layerB = Input(shape=(n_data.shape[1:]),name='in_n')
maskA = Masking(mask_value=0.)(input_layerA)
maskB = Masking(mask_value=0.)(input_layerB)

pass1 = simp_enc(out_dim=input_size+25,rate=0.2)(maskA)
pass2 = simp_enc(out_dim=input_size+25,rate=0.2)(maskB)

conc = Concatenate()([pass1,pass2])
flat = tf.keras.layers.Flatten()(conc)


poutpass = simp_enc(80,0.2)(flat)


yhat = Dense(1,activation='softmax')(poutpass)
model = tf.keras.Model(inputs=[input_layerA,input_layerB],outputs=[yhat],name='test_model')
model.summary()

with tf.device('CPU:0'):
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(),optimizer=tf.keras.optimizers.Nadam(),metrics=['accuracy'])
with tf.device('GPU:0'):
    model.fit({"in_p":p_data,"in_n":n_data},labels,batch_size=batch_size,epochs=epochs,shuffle=True,verbose=1,validation_split=0.3)