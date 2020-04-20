import tensorflow as tf
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

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
  image = tf.cast(image, tf.float32)
  # image = tf.image.resize_with_pad(image, 150, 150)
  image = tf.image.resize(image, (150, 150), method='gaussian', preserve_aspect_ratio=False, antialias=True)
  image /= 255.0
  return image, label

train = train.map(scale)
validation = validation.map(scale)
testing = testing.map(scale)

train = train.batch(128)
validation = validation.batch(size_validation)
testing = testing.batch(size_testing)

validation_input, validation_target = next(iter(validation))

# Model
hidden_layer_size = 50
output_size = info.features['label'].num_classes # 2 = malaria or not

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(150,150,3)),
  tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
  tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
  tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
  train,
  epochs=100,
  callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='loss'),
    tf.keras.callbacks.ModelCheckpoint('../models/malaria', save_best_only=True, monitor='loss', verbose=1)
  ],
  validation_data=(validation_input, validation_target),
  validation_steps=1
)

test_loss, test_accuracy = model.evaluate(testing)

print(test_loss, test_accuracy)
