import numpy as np

input_size = 5
num_layers = 2
rate = 0.2

### import data
f1 = np.load('f1.npy')
labels = np.load('labels.npy')
###

# Testing the effectiveness of masking single features to single labels with time series consideration.
# Mask such that N layers Considers N features, But as for now. We consider that N layers maps to N heads, Where each head maps to a single node in the input.
# we ask the heads to condsider the relationship between nodes with respect to the masking function. Such if we have 5 nodes head 1 considers N1,N2,N3,N4,N5 
# head 2 considers N1,N2,N3,N4,N05. head 3 considers N1,N2,N3,N04,N05. etc. We say that given some input X of magnitude M. The amount of heads should be M-1.

"""
    [A],[B],[C],[D],[E] , M-1 = 4, N=4

    L1 , ABCDE -> P1
    L2 , ABCD0 -> P2 -> [RANDOM FOREST
    L3 , ABC00 -> P3 -> TRANSFORMATION]  
    L4 , AB000 -> P4

    Force partial differential consideration over the entire time series. Then using collective reasoning consider the forward sequence using random forest transformations of 
    the previous layer. |Pn>. 

"""
import tensorflow as tf

f1 = np.load('f1.npy')
labels = np.load('labels.npy')

input_layer = tf.keras.layers.Input(shape=(input_size,))

form = [[True,True,True,True,True],[True,True,True,True,True]]
att_layer = tf.keras.layers.Attention()([input_layer,input_layer],mask=form)
df = tf.keras.layers.Dense(1,activation='softmax')(att_layer)

model = tf.keras.models.Model(inputs=input_layer,outputs=df)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(f1,labels,epochs=2,batch_size=input_size,validation_split=0.2,verbose=1,shuffle=True)
