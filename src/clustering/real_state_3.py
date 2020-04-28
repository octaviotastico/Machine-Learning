from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import numpy as np
sns.set()

# size_neighbour_1 = np.random.normal(30, 10, 3000) + 10
# size_neighbour_2 = np.random.normal(50, 30, 3000) + 10
# size_neighbour_3 = np.random.normal(40, 30, 3000) + 10
# all_sizes = np.concatenate([size_neighbour_1, size_neighbour_2, size_neighbour_3])

# d_to_Tokio_neighbour_1 = np.random.normal(5, 3, 3000)
# d_to_Tokio_neighbour_2 = np.random.normal(10, 2, 3000)
# d_to_Tokio_neighbour_3 = np.random.normal(15, 2, 3000)
# all_distances = np.concatenate([d_to_Tokio_neighbour_1, d_to_Tokio_neighbour_2, d_to_Tokio_neighbour_3])

# price_neighbour_1 = size_neighbour_1 * 25000 + d_to_Tokio_neighbour_1 + np.random.normal(5000000, 1000000, 3000)
# price_neighbour_2 = size_neighbour_2 * 25000 + d_to_Tokio_neighbour_2 + np.random.normal(2500000, 500000, 3000)
# price_neighbour_3 = size_neighbour_3 * 7000 + d_to_Tokio_neighbour_3 + np.random.normal(500000, 750000, 3000)
# all_prices = np.concatenate([price_neighbour_1, price_neighbour_2, price_neighbour_3])

# cluster_tag = np.concatenate([np.full(3000, 0), np.full(3000, 1), np.full(3000, 2)])
# dataframe = pd.DataFrame({'house_size': np.absolute(all_sizes), 'km_to_Tokio': np.absolute(all_distances), 'house_price': np.absolute(all_prices), 'clusters': cluster_tag})

dataset_dir = '../../datasets'

# dataframe.to_csv(f'{dataset_dir}/real_state_3d_2.csv')

dataset = pd.read_csv(f'{dataset_dir}/real_state_3d_2.csv')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Original Scatterplot')
ax.scatter(dataset['house_size'], dataset['km_to_Tokio'], dataset['house_price'])
ax.set_xlabel('House Size')
ax.set_ylabel('KM To Tokio')
ax.set_zlabel('House Price')
plt.show()

kmeans = KMeans(3)
preprocess = preprocessing.scale(dataset.drop(['clusters'], axis=1))
predicted_clusters = kmeans.fit_predict(preprocess)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Predicted Scatterplot')
ax.scatter(dataset['house_size'], dataset['km_to_Tokio'], dataset['house_price'], c=predicted_clusters, cmap='rainbow')
ax.set_xlabel('House Size')
ax.set_ylabel('KM To Tokio')
ax.set_zlabel('House Price')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Real Scatterplot')
ax.scatter(dataset['house_size'], dataset['km_to_Tokio'], dataset['house_price'], c=dataset['clusters'], cmap='rainbow')
ax.set_xlabel('House Size')
ax.set_ylabel('KM To Tokio')
ax.set_zlabel('House Price')
plt.show()