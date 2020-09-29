import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#loading dataset from the seaborn
iris= sns.load_dataset('iris')

#vizualing the data
sns.pairplot(data= iris)
setosa= iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_length'], setosa['sepal_width'], cmap= 'plasma', shade= True, shade_lowest= False)

#Train-Test split 
from sklearn.model_selection import train_test_split
x= iris.drop('species', axis=1)
y= iris['species']
xtrain, xtest, ytrain, ytest= train_test_split(x,y, test_size= 0.33, random_state= 106)

#using support vector machine algo
from sklearn.svm import SVC
svmodel1= SVC()
svmodel1.fit(xtrain, ytrain)
pred1= svmodel1.predict(xtest)

#model evalution
from sklearn.metrics import classification_report, confusion_matrix
mat1= confusion_matrix(ytest, pred1)
rep1= classification_report(ytest, pred1)

#using gridsearch
from sklearn.model_selection import GridSearchCV
param_grid= {'C':[0.1,1,10,100,1000], 'gamma':[0.1,0.001,0.0001,1]}
model2= GridSearchCV(SVC(), param_grid, verbose= 3)
model2.fit(xtrain,ytrain)
model2.best_params_
model2.best_estimator_

model2.fit(xtrain, ytrain)
pred2= model2.predict(xtest)

mat2= confusion_matrix(ytest, pred2)
rep2= classification_report(ytest, pred2)

