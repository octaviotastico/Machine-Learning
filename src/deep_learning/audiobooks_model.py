import numpy as np
import tensorflow as tf

dataset_dir = '../../datasets/'

# Loading Data
npz = np.load(f'{dataset_dir}/audiobooks_cleaned_train.npz')
training_inputs = npz['inputs'].astype(np.float)
training_targets = npz['targets'].astype(np.int)

npz = np.load(f'{dataset_dir}/audiobooks_cleaned_validation.npz')
validation_inputs = npz['inputs'].astype(np.float)
validation_targets = npz['targets'].astype(np.int)

npz = np.load(f'{dataset_dir}/audiobooks_cleaned_testing.npz')
testing_inputs = npz['inputs'].astype(np.float)
testing_targets = npz['targets'].astype(np.int)

# Model
input_size = 10
output_size = 2
hidden_layers_size = 50

model = tf.keras.Sequential([
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
  training_inputs,
  training_targets,
  validation_data=(validation_inputs, validation_targets),
  callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],
  validation_steps=1,
  batch_size=100,
  epochs=100,
  verbose=2
)

test_loss, test_accuracy = model.evaluate(testing_inputs, testing_targets)
print(test_loss, test_accuracy)
