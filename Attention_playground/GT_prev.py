import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Flatten,MultiHeadAttention



##### transformer layer definitions #####

dnn = 64
drop_rate = 0.2

feedforward = tf.keras.Sequential([
	Dense(dnn,activation='relu'),
	Dense(dnn,activation='relu'),

	],name="feedforward")

class gt_model(tf.keras.Model):
	def train_step(self,data):
		x,y = data

		with tf.GradientTape() as tape:
			y_pred = self(x,training=True)# preform predictions
			loss = self.compiled_loss(y,y_pred,regularization_losses=self.losses) # compute difference between true and predicted values

		trainable_vars = self.trainable_variables # establish all trainable variables
		gradients = tape.gradient(loss,trainable_vars) # preform autograd to calculate adjusted new state of variables

		self.optimizer.apply_gradients(zip(gradients,trainable_vars)) # apply change calculated
		self.compiled_metrics.update_state(y,y_pred) # calculate new metrics and update
		return {m.name: m.result() for m in self.metrics}

xp = np.load("p_multi.npy")
y = np.load("labels.npy")



inputs = tf.keras.Input(shape=(xp.shape[1:]))
ff = feedforward(inputs)
flat = Flatten()(ff)

outputs = tf.keras.layers.Dense(1,activation='softmax')(flat)



model = gt_model(inputs, outputs)
model.compile(optimizer="nadam", loss="BinaryFocalCrossentropy", metrics=["accuracy"])
model.summary()

model.fit(xp, y, epochs=3)