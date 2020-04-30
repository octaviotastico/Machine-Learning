import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Data
def preprocess(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (300, 300), method='gaussian', preserve_aspect_ratio=False, antialias=True)
  image = tf.image.rgb_to_grayscale(image)
  image /= 255.0
  return image, label

train = tfds.load('cats_vs_dogs', as_supervised=True, split='train[:80%]').map(preprocess).batch(32)
validation = tfds.load('cats_vs_dogs', as_supervised=True, split='train[80%:90%]').map(preprocess).batch(2326)
testing = tfds.load('cats_vs_dogs', as_supervised=True, split='train[90%:]').map(preprocess).batch(32)

a = 0
for _ in testing:
  a+=1
print(a)

validation_input, validation_target = next(iter(validation))

# Model
output_size = 2
hidden_layers_size = 50

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(300, 300, 1)),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
  train,
  epochs=100,
  # callbacks=[
    # tf.keras.callbacks.EarlyStopping(patience=2),
    # tf.keras.callbacks.ModelCheckpoint('../../models/cats_vs_dogs', save_best_only=True, monitor='loss', verbose=1)
  # ],
  validation_data=(validation_input, validation_target),
  validation_steps=2326
)

test_loss, test_accuracy = model.evaluate(testing)
print(test_loss, test_accuracy)