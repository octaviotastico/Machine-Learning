import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
sns.set()

dataset_dir = '../../datasets/'

def create_and_print_kmeans(n):
  kmeans = KMeans(n)
  sepal_scaled = data.iloc[:,:]
  sepal_scaled = preprocessing.scale(sepal_scaled)
  data['Clusters'] = kmeans.fit_predict(sepal_scaled)
  # kmeans.inertia_ gives the wcss score, to help you choose the amount of clusters

  plt.scatter(data['sepal_length'],data['sepal_width'],c=data['Clusters'],cmap='rainbow')
  plt.title(f'Data clustering with {n} clusters')
  plt.xlabel('Sepal Length')
  plt.ylabel('Sepal Width')
  plt.show()

data = pd.read_csv(f'{dataset_dir}/iris_flower_dataset.csv')
print('Original data is:', data)

for i in range(6):
  create_and_print_kmeans(i)

targets = pd.read_csv(f'{dataset_dir}/iris_flower_dataset_targets.csv')
targets['species'] = targets['species'].map({'setosa':0,'versicolor':1,'virginica':2})

plt.scatter(targets['sepal_length'],targets['sepal_width'],c=targets['species'],cmap='rainbow')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
