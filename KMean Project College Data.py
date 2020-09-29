import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading cvs file 
data= pd.read_csv('College_Data')
data.info()
data.desribe()

#Vizualing the data
sns.scatterplot('Room.Board','Grad.Rate', hue= 'Private', data= data, cmap= 'viridis')
sns.set_style('whitegrid')
sns.scatterplot('F.Undergrad', 'Outstate', data= data, hue= 'Private')

fig1= sns.FacetGrid(data, hue= 'Private', size= 6)
fig1= fig1.map(plt.hist, 'Outstate', bins= 30, alpha= 0.6)

fig2= sns.FacetGrid(data, hue= 'Private', size= 4, aspect= 2)
fig2= fig2.map(plt.hist, 'Grad.Rate', bins= 30, alpha= 0.5)

#As one in one of the input have Grad.Rate greater than 100 which don't make sence
g= data[data['Grad.Rate']>100]
data.loc[95, 'Grad.Rate']= 100
data.drop(19, axis= 1, inplace= True)

#Creating the label of the data using kmean cluster
from sklearn.cluster import KMeans
Kmodel= KMeans(n_clusters= 2)
Kmodel.fit(data.drop(['Private', 'Unnamed: 0'], axis= 1))
Kmodel.cluster_centers_

def cluster(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0
data['Cluster']= data['Private'].apply(cluster)

#How well our model works?
from sklearn.metrics import confusion_matrix, classification_report
mat= confusion_matrix(data['Cluster'], Kmodel.labels_)
rep= classification_report(data['Cluster'], Kmodel.labels_)



