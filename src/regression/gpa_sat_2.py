import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set()

dataset_dir = '../../datasets/'

dataset = pd.read_csv(f'{dataset_dir}/gpa_sat.csv')

print('Let\'s peek into the dataset:')
print(dataset)
print(dataset.describe())

x, y = dataset['SAT'].to_numpy(), dataset['GPA']

plt.scatter(x, y)
plt.title('Original Data')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

x = x.reshape(-1, 1)
sk_reg = LinearRegression().fit(x, y)
print('Regression Score:', sk_reg.score(x, y))

a = sk_reg.coef_
b = sk_reg.intercept_

function = a*x + b

plt.scatter(x, y)
fig = plt.plot(x, function, lw=3, c='red', label ='regression')
plt.title('Original Data and Regression')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

print('Predicting GPA with STA = 1555:', sk_reg.predict([[1555]]))
print('Predicting GPA with STA = 1555:', a*(1555)+b)