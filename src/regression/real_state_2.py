import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
sns.set()

# house_size = np.random.uniform(500, 1500, 300)
# house_price = house_size * 250 * np.random.uniform(0.1, 2) + np.random.uniform(-10000, 10000, 300)
# km_to_Tokio = (np.random.uniform(5000000, 15000000, 300) / house_price)

# dataframe = pd.DataFrame({'house_size': house_size, 'km_to_Tokio': km_to_Tokio, 'house_price': house_price})
# dataframe.to_csv(f'{dataset_dir}/real_state_3d.csv')

dataset_dir = '../../datasets'

dataset = pd.read_csv(f'{dataset_dir}/real_state_3d.csv')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataset['house_size'], dataset['km_to_Tokio'], dataset['house_price'])
ax.set_xlabel('House Size')
ax.set_ylabel('KM To Tokio')
ax.set_zlabel('House Price')
plt.show()

reg = LinearRegression().fit(np.column_stack((dataset['house_size'], dataset['km_to_Tokio'])), dataset['house_price'])
a, b = reg.coef_, reg.intercept_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataset['house_size'], dataset['km_to_Tokio'], dataset['house_price'])
ax.set_xlabel('House Size')
ax.set_ylabel('KM To Tokio')
ax.set_zlabel('House Price')

ax.plot_surface(
  np.array([[400, 400], [1500, 1500]]),
  np.array([[30, 255], [30, 255]]),
  reg.predict(np.array([[400, 400, 1500, 1500], [30, 255, 30, 255]]).T).reshape((2, 2)),
  alpha=.5
)

plt.show()