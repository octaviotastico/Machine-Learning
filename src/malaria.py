import tensorflow as tf
import tensorflow_datasets as tfds

# train = tfds.load('malaria', as_supervised=True, split='train[:80%]')
# validation = tfds.load('malaria', as_supervised=True, split='train[80%:90%]')
# testing = tfds.load('malaria', as_supervised=True, split='train[90%:]')
dataset, info = tfds.load('malaria', as_supervised=True, with_info=True)
dataset = dataset['train'].shuffle(10000)

size_dataset = raw_data = info.splits['train'].num_examples
size_training = int(size_dataset * 0.8)
size_validation = int(size_dataset * 0.1)
size_testing = size_dataset - size_training - size_validation

train = dataset.take(size_training)
validation = dataset.skip(size_training).take(size_validation)
testing = dataset.skip(size_training+size_validation).take(size_testing)

def scale(image, label):
  print(image)
  image = tf.cast(image, tf.float32)
  # image = tf.image.resize_with_pad(image, 150, 150)
  image = tf.image.resize(image, (150, 150), method='gaussian', preserve_aspect_ratio=False, antialias=True)
  image /= 255.0
  print(image)
  return image, label

train = train.map(scale)
validation = validation.map(scale)
testing = testing.map(scale)

train = train.batch(100)
validation = validation.batch(size_validation)
testing = testing.batch(size_testing)

validation_input, validation_target = next(iter(validation))


