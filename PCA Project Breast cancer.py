import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading dataset
from sklearn.datasets import load_breast_cancer
cancer= load_breast_cancer()
cancer_df= pd.DataFrame(cancer['data'], columns= cancer['feature_names'])
 
#Scaling the values before applying pca
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(cancer_df)
scaled_feat= scaler.transform(cancer_df)

#Calling PCA algorithm for the scaled data
from sklearn.decomposition import PCA
pca= PCA(n_components= 2)
pca.fit(scaled_feat)
pca_comp= pca.transform(scaled_feat)

#Plotting these two component that we just got using PCA algorithm
plt.figure(figsize= (8,6))
sns.set_style('whitegrid')
plt.scatter(pca_comp[:,0], pca_comp[:,1], c=cancer['target'], cmap= 'plasma')
plt.xlabel('First component')
plt.ylabel('Second component')

#vizualing the components
pca.components_
df_comp= pd.DataFrame(data= pca.components_, columns= cancer['feature_names'])
sns.heatmap(df_comp, cmap= 'plasma')
