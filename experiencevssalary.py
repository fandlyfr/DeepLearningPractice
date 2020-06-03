import tensorflow as tf

import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# #define data
# years   = np.array([1,2,3,4,5,6,7,8,9,10],  dtype=int)
# salary = np.array([2,4,6,8,10,12,14,16,18,20],  dtype=int)

#try celcius to kelvin
years   = np.array([1,2,3,4,5],  dtype=int)
salary = np.array([274,275,276,277,278],  dtype=int)

for i,c in enumerate(years):
    print("{} Years Experience = {} Salary".format(years[i],salary[i]))


l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

#bisajuga seperti ini
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1, input_shape=[1])
# ])

model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(2))

#train
history = model.fit(years,salary,epochs=1000,verbose=False)
print("Finished Training the model")

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
# plt.show()
print(model.predict([1]))

# print("These are the layer variables: {}".format(l0.get_weights()))