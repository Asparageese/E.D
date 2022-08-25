from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Masking, GRU, Bidirectional, Dense, Flatten, GaussianDropout, LayerNormalization, Concatenate,BatchNormalization,AdditiveAttention


class BIG(tf.keras.layers.Layer):
    def __init__(self, outputs):
        super(BIG, self).__init__()
        self.outputs = outputs

        self.qbig1 = Bidirectional(
            GRU(self.outputs, return_sequences=True, activation="tanh", recurrent_activation="sigmoid",
                recurrent_dropout=0.0, unroll=False, use_bias=True), merge_mode='concat')

        self.drop1 = GaussianDropout(0.2)
        self.norm1 = BatchNormalization()

        self.vbig1 = Bidirectional(
            GRU(self.outputs, return_sequences=True, activation="tanh", recurrent_activation="sigmoid",
                recurrent_dropout=0.0, unroll=False, use_bias=True), merge_mode='concat')

        self.att1 = AdditiveAttention()
        self.norm1 = LayerNormalization()

        self.bigo = Bidirectional(
            GRU(self.outputs, activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.0, unroll=False,
                use_bias=True))

        self.query = Dense(input_size, activation='relu')
        self.value = Dense(prediction_entropy,activation='relu')

    def call(self, inputs):
        q = self.qbig1(inputs)
        x = self.drop1(q)
        x = self.norm1(x)
        v = self.vbig1(x)
        a = self.att1([q, v])
        x = self.bigo(a)
        return self.query(x), self.value(x)




batch_size = 64
epochs = 3
prediction_entropy = 2

### import data
p_data = np.load('p_set.npy')
n_data = np.load('n_set.npy')
labels = np.load('labels.npy')
input_size = 30
p_data, n_data = p_data[:,:,tf.newaxis], n_data[:,:,tf.newaxis]

p_data,n_data, labels = shuffle(p_data,n_data,labels,random_state=0)

input_layerA = Input(shape=(p_data.shape[1:]),name='in_p')
input_layerB = Input(shape=(n_data.shape[1:]),name='in_n')
maskA = Masking(mask_value=0.)(input_layerA)
maskB = Masking(mask_value=0.)(input_layerB)

pq,pv = BIG(32)(maskA)
nq,nv = BIG(32)(maskB)

att1 = AdditiveAttention()([pq,nq])
att2 = AdditiveAttention()([pv,nv])

conc = Concatenate()([att1,att2])

yhat = Dense(1,activation='softmax')(conc)
model = tf.keras.Model(inputs=[input_layerA,input_layerB],outputs=[yhat],name='test_model')
model.summary()

with tf.device('CPU:0'):
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(),optimizer=tf.keras.optimizers.Nadam(),metrics=['accuracy'])
with tf.device('GPU:0'):
    model.fit({"in_p":p_data,"in_n":n_data},labels,batch_size=batch_size,epochs=epochs,shuffle=True,verbose=1,validation_split=0.3)