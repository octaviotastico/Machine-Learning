import tensorflow as tf
import tensorflow_datasets as tfds

# Data
mnist, info = tfds.load('mnist', with_info=True, as_supervised=True)
train, test = mnist['train'].shuffle(10000), mnist['test']

size_train = info.splits['train'].num_examples
size_validation = tf.cast(0.1 * size_train, tf.int64)
size_test = tf.cast(info.splits['test'].num_examples, tf.int64)

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255.0
  return image, label

scaled_train = train.map(scale)
scaled_test = test.map(scale)

train = scaled_train.skip(size_validation)
validation = scaled_train.take(size_validation)

train = train.batch(100)
validation = validation.batch(size_validation)
test = test.batch(size_test)

validation_input, validation_target = next(iter(validation))

# Model
output_size = 10
hidden_layers_size = 50

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28,1)), # input_size = 28*28*1 = 784
  tf.keras.layers.Dense(hidden_layers_size, activation='relu'),
  tf.keras.layers.Dense(hidden_layers_size, activation='relu'),
  tf.keras.layers.Dense(hidden_layers_size, activation='relu'),
  tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

model.fit(
  train,
  epochs=100,
  callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint('../../models/mnist', save_best_only=True, verbose=1)
  ],
  validation_data=(validation_input, validation_target),
  validation_steps=1,
  verbose=2
)

test_loss, test_accuracy = model.evaluate(test)
