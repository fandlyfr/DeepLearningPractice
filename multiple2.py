import tensorflow as tf

import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# #define data
#multiple 2
a   = np.array([1,2,3,4,5],  dtype=int)
b = np.array([274,275,276,277,278],  dtype=int)

for i,c in enumerate(a):
    print("{} * 2 = {}".format(a[i],b[i]))


l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

#bisajuga seperti ini
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1, input_shape=[1])
# ])

model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(2))

#train
history = model.fit(a,b,epochs=1000,verbose=False)
print("Finished Training the model")

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
# plt.show()
print(model.predict([1]))

# print("These are the layer variables: {}".format(l0.get_weights()))