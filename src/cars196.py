import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Data
dataset, info = tfds.load('cars196', with_info=True, as_supervised=True)
train, test = dataset['train'], dataset['test']

size_train = info.splits['train'].num_examples
size_validation = tf.cast(0.05 * size_train, tf.int64)
size_test = tf.cast(info.splits['test'].num_examples, tf.int64)

first_image = next(iter(train))
image = first_image[0].numpy()
plt.imshow(image)
plt.show()

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255.0
  return image, label

scaled_train = train.map(scale)
scaled_test = test.map(scale)
