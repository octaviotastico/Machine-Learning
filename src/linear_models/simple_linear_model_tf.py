from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.cluster import KMeans
sns.set()

# Random input data to train
observations = 10000
xs = np.random.uniform(10, 10, (observations, 1))
zs = np.random.uniform(10, 10, (observations, 1))

noise = np.random.uniform(-5, 5, (observations, 1))

inputs = np.column_stack((xs, zs))
targets = 3*xs + 5*zs + 7 + noise

# Algorithm
input_size = 2
output_size = 1

init_range = 0.1
random_weight = tf.random_uniform_initializer(-init_range, init_range)
random_bias = tf.random_uniform_initializer(-init_range, init_range)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(output_size,
                        kernel_initializer=random_weight,
                        bias_initializer=random_bias)
])

learning_rate = 0.02
custom_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

model.fit(inputs, targets, epochs=100, verbose=2)

weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]

print('Final weights and biases:', weights, bias)

plt.plot(model.predict_on_batch(inputs).round(1), targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()
