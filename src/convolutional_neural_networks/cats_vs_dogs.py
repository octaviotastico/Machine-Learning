import tensorflow as tf
import tensorflow_datasets as tfds

# Data
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)
data = dataset['train'].shuffle(10000)

train = tfds.load('cats_vs_dogs', as_supervised=True, split='train[:80%]')
validation = tfds.load('cats_vs_dogs', as_supervised=True, split='train[80%:90%]')
testing = tfds.load('cats_vs_dogs', as_supervised=True, split='train[90%:]')

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255.0
  return image, label

# Model
output_size = 2
hidden_layers_size = 50
