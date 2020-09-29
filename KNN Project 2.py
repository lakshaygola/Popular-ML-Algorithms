#Reading csv filemport pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

df= pd.read_csv('KNN_Project_Data')
df.head()

#Pair plot in order to understand the relationship between the features
sns.pairplot(data= df)
plt.tight_layout()

#Standardize the features
from sklearn.preprocessing import StandardScaler
Scale = StandardScaler()
Scale.fit(df.drop('TARGET CLASS', axis= 1))
scaled_feat= pd.DataFrame(Scale.transform(df.drop('TARGET CLASS' , axis= 1)), columns= df.columns[:-1] )

#Train-Test split
from sklearn.model_selection import train_test_split
x= df[['XVPM', 'GWYH', 'TRAT', 'TLLZ', 'IGGA', 'HYKR', 'EDFS', 'GUUB', 'MGJM','JHZC']]
y= df['TARGET CLASS']
xtrain, xtest, ytrain, ytest= train_test_split(x,y,test_size= 0.39, random_state=20)

#Appling KNN algorithms
from sklearn.neighbors import KNeighborsClassifier
df_knn= KNeighborsClassifier(n_neighbors= 1)
df_knn.fit(xtrain, ytrain)

#Predicting the values
pred= df_knn.predict(xtest)

#Confusion matrix & Classification report
from sklearn.metrics import confusion_matrix, classification_report
mart= confusion_matrix(ytest, pred)
rep= classification_report(ytest, pred)

#Finding proper k-values for the model using elbow method
error= []

for i in range(1,40):
    kn = KNeighborsClassifier(n_neighbors = i)
    kn.fit(xtrain,ytrain)
    pred_i= kn.predict(xtest)
    error.append(np.mean(pred_i != ytest))
  
#Plot the graph for the k-value having lowest error
plt.figure(figsize=(10,8))  
plt.plot(range(1,40), error, color= 'red' , linestyle= ':', marker= '^', markerfacecolor= 'red', markersize= 9)
plt.title('K-values vs Error')
plt.xlabel('K-Values')
plt.ylabel('Error')

#K-values = 26 is suitable for this model
df_knn= KNeighborsClassifier(n_neighbors= 26)
df_knn.fit(xtrain, ytrain)  
pred_b= df_knn.predict(xtest)
mart= confusion_matrix(ytest, pred_b)
rep= classification_report(ytest, pred_b)
