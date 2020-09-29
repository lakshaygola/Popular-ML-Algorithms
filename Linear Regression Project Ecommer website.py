import pandas as pd
import numpy as np
import matplotlib as mlt
import seaborn as sns

#Reading csv file which contain data
data = pd.read_csv('Ecommerce Customers')

#Exploring data 
data.head()
data.info()
data.describe()

#Time on website vs yearly amount spent
sns.jointplot('Time on Website', 'Yearly Amount Spent', data)

#Time on app vs yearly amount spent
sns.jointplot('Time on App', 'Yearly Amount Spent', data)

#Time on app vs Lenght of membership
sns.jointplot('Time on App' , 'Length of Membership' , data , kind = 'hex')

sns.pairplot(data)
#From the pair plot we get that lenght of membership is most correlated with yearly amount spent

#linear model plot of yearly amount spent vs lenght of membership
sns.lmplot('Yearly Amount Spent' , 'Length of Membership' ,data)

#Training test split of data
from sklearn.model_selection import train_test_split
data.columns
X = data[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

y = data['Yearly Amount Spent']

X_train , X_test , y_train , y_test = train_test_split(X , y, test_size = 0.3 , random_state = 101)

#training the model 
from sklearn.linear_model import LinearRegression
ll = LinearRegression()
ll.fit(X_train , y_train)

#Values of intercept and coefficient
print(ll.coef_)
print(ll.intercept_)

#Predicting the test data
prediction = ll.predict(X_test)

#Scatter plot
sns.scatterplot(y_test , prediction)

#Calculating Error 
from sklearn import metrics as m
print('MAE : ' , m.mean_absolute_error(y_test, prediction))
print('MSE : ' , m.mean_squared_error(y_test, prediction))
print('RMSE : ' , sqrt(m.mean_squared_error(y_test, prediction)))

#Exploring the residuals
sns.distplot((y_test-prediction), bins = 50 )
sns.set_style('whitegrid')

#How much each feature related with the sales of the company
coefficient = pd.DataFrame(ll.coef_ , X.columns)
coefficient.columns = ['Coefficients']


#Real world data set
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston
bostondf = pd.DataFrame(boston.data , columns = boston.feature_names)




