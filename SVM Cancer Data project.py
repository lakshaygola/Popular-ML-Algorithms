import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the data from the sklearn dataset
from sklearn.datasets import load_breast_cancer
cancer= load_breast_cancer()

#making Dataframe form the dictinary
cancer_reports= pd.DataFrame(data= cancer['data'], columns= cancer['feature_names'])
cancer_result= pd.DataFrame(data= cancer['target'], columns= ['Cancer'])

#Train-Test split of the data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest= train_test_split(cancer_reports, np.ravel(cancer_result), test_size= 0.30, random_state= 101)

#Creating the model
from sklearn.svm import SVC
model1= SVC()
model1.fit(xtrain, ytrain)

pred1= model1.predict(xtest)

#Analyzing the model prediction
from sklearn.metrics import confusion_matrix, classification_report
mat1= confusion_matrix(ytest,pred1)
rep1= classification_report(ytest,pred1)

#using the grid search method
from sklearn.model_selection import GridSearchCV
para_grid= {'C':[0.1,0,1,10,100,1000], 'gamma':[0.1,1,2,0.001,0.001]}
model2= GridSearchCV(SVC(), para_grid, verbose=3)
model2.fit(xtrain,ytrain)
model2.best_params_
pred2= model2.predict(xtest)

mat2= confusion_matrix(ytest, pred2)
rep2= classification_report(ytest, pred2)
