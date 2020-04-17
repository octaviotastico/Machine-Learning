import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import os
sns.set()

# Function to print the logistic regression
def log_reg_f(x, a, b):
  return np.array(np.exp(a*x+b) / (1+np.exp(a*x+b)))

# Original data
data = pd.read_csv('../datasets/bank_data.csv')
data = data.drop(['Unnamed: 0'], axis = 1)
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Values of the scatter plot
x = data['interest_rate'] # data[['interest_rate', 'duration']]
y = data['y']

# Creating the regression
x = sm.add_constant(x)
result = sm.Logit(y,x).fit()
result.summary()

x_f = np.sort(np.array(x['interest_rate']))
y_f = np.sort(log_reg_f(x_f, result.params[0], result.params[1]))

plt.scatter(x['interest_rate'], y)
plt.show()