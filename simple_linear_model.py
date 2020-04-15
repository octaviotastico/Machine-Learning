import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
sns.set()

# Random input data to train
observations = 10000
xs = np.random.uniform(low=-10, high=10, size=(observations, 1)) # Size = NxK (n of observations times number of variables)
zs = np.random.uniform(low=-10, high=10, size=(observations, 1))

inputs = np.column_stack((xs, zs))

# Create targets we will aim at
# Targets = f(x,z) = 3*x + 5*z + 7 + noise
noise = np.random.uniform(-5, 5, (observations, 1))
targets = 3*xs + 5*zs + 7 + noise

# Plot training data
def plot_training_data():
    targets = targets.reshape(observations,)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot(xs, zs, targets)
    ax.set_xlabel('xs')
    ax.set_ylabel('zs')
    ax.set_zlabel('Targets')
    ax.view_init(azim = 100)
    plt.show()
    targets = targets.reshape(observations, 1)

# Algorithm
init_range = 0.1

weights = np.random.uniform(-init_range, init_range, (2, 1)) # Size is (2,1) cos we have 2 variables and 1 output
biases = np.random.uniform(-init_range, init_range, 1) # Size is 1 because we have only 1 output Y
print('Random weights and biases:', weights, biases)

learning_rate = 0.03

# Training
old_loss = 0
for _ in range (1000):
    outputs = np.dot(inputs, weights) + biases
    deltas = outputs - targets
    loss = np.sum(deltas ** 2) / 2 / observations # L2-Norm loss formula
    if (abs(old_loss - loss) < 0.000000000001):
        break

    old_loss = loss
    deltas_scaled = deltas / observations
    weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
    biases = biases - learning_rate * np.sum(deltas_scaled)

print('Final weights and biases:', weights, biases)

plt.plot(outputs, targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()
