from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Masking, Dense, Dropout, BatchNormalization, Flatten

class fnn(tf.keras.layers.Layer):
    def __init__(self,out_dim):
        super(fnn, self).__init__()
        self.out_dim = out_dim
        self.d1 = Dense(self.out_dim,activation='tanh')
        self.d2 = Dense(self.out_dim,activation='tanh')

    def call(self, inputs):
        x = self.d1(inputs)
        return self.d2(x)

class encoder_layer(tf.keras.layers.Layer):
    def __init__(self,out_dim,rate):
        super(encoder_layer, self).__init__()
        self.out_dim = out_dim
        self.rate = rate
        self.fnn = fnn(self.out_dim)
        self.drop = Dropout(self.rate)
        self.norm = BatchNormalization()

    def call(self, inputs):
        x = self.fnn(inputs)
        x = self.drop(x)
        return self.norm(x)


class encoder(tf.keras.layers.Layer):
    def __init__(self, out_dim, rate):
        super(encoder, self).__init__()
        self.out_dim = out_dim
        self.rate = rate

        self.mask = Masking(mask_value=0.)
        self.enc_layers = [encoder_layer(out_dim=self.out_dim,rate=self.rate) for _ in range(2)]
        self.drop = Dropout(self.rate)
    def call(self,pos,neg):
        join = [pos,neg]
        for i in range(2):
            x = self.drop(join[i])
            x = self.mask(x)
            return self.enc_layers[i](x)



batch_size = 32
epochs = 6


### import data
p_data = np.load('p_multi.npy')
n_data = np.load('n_multi.npy')
labels = np.load('labels.npy')

#p_data, n_data = p_data[:,:,tf.newaxis], n_data[:,:,tf.newaxis]

p_data,n_data, labels = shuffle(p_data,n_data,labels,random_state=0)

pos_layer = Input(shape=(p_data.shape[1:]))
neg_layer = Input(shape=(n_data.shape[1:]))

encoder_mechanisim = encoder(25,0.3)(pos_layer,neg_layer)
flat = Flatten()(encoder_mechanisim)

y_hat = Dense(1,activation='softmax')(flat)

model = tf.keras.models.Model(inputs=[pos_layer,neg_layer],outputs=[y_hat],name='CPU_DNN_MASKING')
model.summary()

model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(), optimizer=tf.keras.optimizers.Nadam(),metrics=['accuracy','mse','mae'])
with tf.device('/GPU:0'):
    model.fit([p_data,n_data],labels,epochs=epochs,batch_size=batch_size,shuffle=True,validation_split=0.3)
