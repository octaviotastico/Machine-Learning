import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Data
dataset, info = tfds.load('cars196', with_info=True, as_supervised=True)
train, test = dataset['train'].shuffle(5000), dataset['test']

size_train = info.splits['train'].num_examples
size_validation = tf.cast(0.05 * size_train, tf.int64)
size_test = tf.cast(info.splits['test'].num_examples, tf.int64)

