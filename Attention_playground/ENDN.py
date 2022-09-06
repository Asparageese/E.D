import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers import Input,Dense,Dropout,BatchNormalization,LayerNormalization,MultiHeadAttention,Flatten,Concatenate

############# Layer defintions #############


feedforward = tf.keras.Sequential([
	Dense(dff,activation='relu')
	Dense(dff,activation='relu')
	],name='feedforward')
