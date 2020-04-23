import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
sns.set()

dataset_dir = '../../datasets/'

dataset = pd.read_csv(f'{dataset_dir}/real_state.csv')

x = np.array(dataset['Size'])
y = np.array(dataset['Price'])

plt.scatter(x, y, c='orange')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

cl = KMeans(n_clusters=2)
clusters = cl.fit_predict(dataset)

ones = np.ones(x.shape)
reg = LinearRegression().fit(np.column_stack((x, ones)), y)
a, b = reg.coef_[0], reg.intercept_
function = a*x+b

plt.scatter(x, y, c=clusters)
plt.xlabel('Size')
plt.ylabel('Price')
plt.plot(x, function, c='red')
plt.show()
