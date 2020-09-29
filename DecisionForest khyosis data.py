import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading the data 
khy = pd.read_csv('kyphosis.csv')
khy.head()

#Exploring the data 
sns.pairplot(khy, hue= 'Kyphosis')

#Train-Test split of the data
from sklearn.model_selection import train_test_split
x= khy[['Age', 'Number', 'Start']]
y= khy['Kyphosis']
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size= 0.3, random_state= 105) 
from sklearn.tree import DecisionTreeClassifier
DTC= DecisionTreeClassifier()
DTC.fit(xtrain, ytrain)
pred= DTC.predict(xtest)

#taking the accuracy from the model that we built
from sklearn.metrics import confusion_matrix, classification_report
mat= confusion_matrix(ytest, pred)
report= classification_report(ytest, pred)

#Now we are using Random forest methofd to predict the same data
from sklearn.ensemble import RandomForestClassifier
RFC= RandomForestClassifier(n_estimators= 100)
RFC.fit(xtrain,ytrain)
pred_r = RFC.predict(xtest)
mat_r= confusion_matrix(ytest, pred_r)
report_r= classification_report(ytest, pred_r)
