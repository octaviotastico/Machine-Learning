import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
sns.set()

dataset_dir = '../../datasets/'

dataset = pd.read_csv(f'{dataset_dir}/gpa_sat.csv')

print('Let\'s peek into the dataset:')
print(dataset)
print(dataset.describe())

x, y = dataset['SAT'], dataset['GPA']

plt.scatter(x, y)
plt.title('Original Data')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

x_reg = sm.add_constant(x)
sm_reg = sm.OLS(y, x_reg).fit()

a = sm_reg.params[1]
b = sm_reg.params[0]

function = a*x + b

plt.scatter(x, y)
fig = plt.plot(x, function, lw=3, c='red', label ='regression')
plt.title('Original Data and Regression')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

x_pred = [1, 1555]
print('Predicting GPA with STA = 1555:', a*(1555)+b)
print('Predicting GPA with STA = 1555:', sm_reg.predict(x_pred)[0])
