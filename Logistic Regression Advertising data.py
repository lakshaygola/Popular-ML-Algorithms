import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as st

#Reading data file
ad_data = pd.read_csv('advertising.csv')
ad_data.describe()
ad_data.info()

#Exlporing the data
sns.set_style('whitegrid')

ad_data['Age'].plot(kind='hist' , bins= 30)
sns.jointplot(x= 'Age', y= 'Area Income' , data= ad_data)
sns.jointplot(x= 'Age', y= 'Daily Time Spent on Site', data= ad_data, kind= 'kdeplot', color= 'red').annotate(st.pearsonr)
sns.jointplot(x= 'Daily Time Spent on Site' , y= 'Daily Internet Usage', data= ad_data, color= 'green').annotate(st.pearsonr)
sns.pairplot(ad_data, hue= 'Cliked on Ad', palette = 'bwr')

#Train-Test split of the data for logistic regression
from sklearn.model_selection import train_test_split 
x= ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y= ad_data['Clicked on Ad']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size= 0.3, random_state= 50)

#Fitting the data in the algorithms
from sklearn.linear_model import LogisticRegression
lrog = LogisticRegression()
lrog.fit(xtrain, ytrain)

#Predicting the feature for the test data
pred = lrog.predict(xtest)

from sklearn.metrics import classification_report
result = classification_report(ytest, pred)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest , pred)
