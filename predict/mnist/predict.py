import matplotlib.image as mplimg
import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess():
  for i in range(10):
    img = Image.open(f'{i}.jpg').convert('L')
    img.save(f'{i}.jpg')

def predict():
  model = tf.keras.models.load_model('../../models/mnist/')
  for i in range(10):
    img = mplimg.imread(f'../../predict/mnist/{i}_.jpg')

    resized = np.array(img)
    print(resized.shape)
    resized.resize((28,28,1))
    print(resized.shape)
    resized = (np.expand_dims(resized, 0))
    print(resized.shape)
    reescaled = resized / 255
    reescaled = -(reescaled - 1)
    prediction = model.predict(reescaled)
    print(f'Prediction for {i} is:', np.argmax(prediction[0]))

preprocess()
predict()