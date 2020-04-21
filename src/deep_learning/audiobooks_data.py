import numpy as np
import pandas as pd
from sklearn import preprocessing

# Loading Data
dataset_dir = '../../datasets/'
raw_data = pd.read_csv(f'{dataset_dir}/audiobooks_data.csv')

# Separating
training = raw_data[ raw_data.columns.difference(['ID', 'Targets']) ].to_numpy()
targets = raw_data['Targets'].to_numpy()

# Scaling the data
training = preprocessing.scale(training)

# Shuffling the data
indices = np.arange(targets.size)
np.random.shuffle(indices)
targets = targets[indices]
training = training[indices]

# Balancing the data
targets_1s = targets.sum()
targets_0s = targets.size - targets_1s

indices_to_remove = []
for i in range(targets.size):
  if len(indices_to_remove) == targets_0s - targets_1s:
    break
  if targets[i] == 0:
    indices_to_remove.append(i)

targets = np.delete(targets, indices_to_remove, axis=0)
training = np.delete(training, indices_to_remove, axis=0)

# Separating Training, Validation and Testing
samples = targets.size
training_samples = int(0.8 * samples)
validation_samples = int(0.1 * samples)
testing_samples = samples - training_samples - validation_samples

training_inputs = training[:training_samples]
training_targets = targets[:training_samples]

validation_inputs = training[training_samples:training_samples + validation_samples]
validation_targets = targets[training_samples:training_samples + validation_samples]

testing_inputs = training[training_samples+validation_samples:]
testing_targets = targets[training_samples+validation_samples:]

np.savez(f'{dataset_dir}/audiobooks_cleaned_train', inputs=training_inputs, targets=training_targets)
np.savez(f'{dataset_dir}/audiobooks_cleaned_validation', inputs=validation_inputs, targets=validation_targets)
np.savez(f'{dataset_dir}/audiobooks_cleaned_testing', inputs=testing_inputs, targets=testing_targets)
