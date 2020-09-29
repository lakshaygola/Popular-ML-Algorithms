import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#Reading the data 
train = pd.read_csv('titanic_train.csv')
train.head()
test = pd.read_csv('titanic_test.csv')

#Exploring the data beforing getting into the classification regression to get the idea of the data set
sns.set_style('whitegrid')
sns.countplot(x= 'Survived' , data = train)
sns.countplot(x= 'Survived' , data = train , hue = 'Sex' )
sns.countplot(x= 'Survived' , data = train , hue = 'Pclass' )

import cufflinks as cf
cf.go_offline()
from plotly.offline import plot
import plotly.graph_objs as go

train['Age'].plot(kind = 'hist' , bins = 40)
train['Fare'].plot(kind = 'hist' , bins = 40)

#Checking for the null values
train.isnull()
plt.figure(figsize = (10,6))
sns.heatmap(train.isnull(), yticklabels = False , cmap = 'coolwarm' ,cbar = False)
sns.heatmap(test.isnull() , yticklabels = False , cmap = 'viridis' , cbar = False)


#Using box plot to get the rough idea about the mean or averge age of each class
plt.figuresize = (10,8)
sns.boxplot(x = 'Pclass', y= 'Age', data = train)
sns.boxplot(x= 'Pclass' , y = 'Age' , data = test)

#Filling up the missing value in age column 
def trainAgenull(col):
    Age = col[0]
    Pclass = col[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else : 
            return 24
    else: return Age

train['Age'] = train[['Age' , 'Pclass']].apply(trainAgenull , axis = 1)

def testAge (col):
    Age = col[0]
    Pclass = col[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 42
        elif Pclass == 2:
            return 26
        elif Pclass == 3:
            return 24
    else:
        return Age
test['Age'] = test[['Age' , 'Pclass']].apply(testAge , axis = 1)

#As the missing values in cabin column is too much so its better to drop the column
train.drop('Cabin' , inplace = True , axis = 1)

test.drop('Cabin' , inplace = True , axis =1)

#Droping all the missing values
train.dropna(inplace = True)

test.dropna(inplace = True)

#Creating the dummy variable for the sex & embarked columns
sex = pd.get_dummies(train['Sex'] , drop_first = True)
sex.head()

embarked= pd.get_dummies(train['Embarked'] , drop_first = True)
embarked.head()

gender = pd.get_dummies(test['Sex'],drop_first = True)
testembarked = pd.get_dummies(test['Embarked'] , drop_first = True)


#Adding these to dataframe into train dataset
train = pd.concat([train , sex , embarked],axis = 1)
train.drop(['Sex' , 'Name' , 'Embarked' , 'Ticket'],inplace = True,axis = 1)

test = pd.concat([test , gender , testembarked] , axis = 1)
test.drop(['Sex' , 'Embarked' , 'Name' , 'Ticket'] , inplace = True , axis = 1)

#Passengerld will not be able to classify the class so we droping that columns
train.drop('PassengerId' , inplace = True , axis =1)
test.drop('PassengerId' , inplace = True , axis = 1)

#Train the model using the training dataset
x = train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q','S']]
y = train['Survived']

xtest = test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']]

from sklearn.linear_model import LogisticRegression
titc = LogisticRegression()
titc.fit(x,y)

#Now predicting the surviver using test data set
perdiction = titc.predict(xtest)

from sklearn.metrics import classification_report


