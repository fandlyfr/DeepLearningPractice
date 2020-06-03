import tensorflow as tf

import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#define data
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
    print("{} degrees Celcius = {} degrees Fahrenheit".format(c,fahrenheit_a[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

#bisajuga seperti ini
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1, input_shape=[1])
# ])

model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.001))

#train
history = model.fit(celsius_q,fahrenheit_a,epochs=1000,verbose=False)
print("Finished Training the model")

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()
print(model.predict([1]))

print("These are the layer variables: {}".format(l0.get_weights()))
