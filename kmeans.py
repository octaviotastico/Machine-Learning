from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

def create_and_print_kmeans(n):
  kmeans = KMeans(n)
  data['Clusters'] = kmeans.fit_predict(data.iloc[:,1:])

  plt.scatter(data['Longitude'],data['Latitude'],c=data['Clusters'],cmap='Pastel2')
  plt.title(f'Data clustering with {n} clusters')
  plt.style.context('Solarize_Light2')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.xlim(-180, 180)
  plt.ylim(-90, 90)
  plt.show()

data = pd.read_csv('./datasets/countries.csv')
print('Original data is:', data)

for i in range(6):
  create_and_print_kmeans(i)
