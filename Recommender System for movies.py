import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#Reading the file 
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
data= pd.read_csv('u.data', sep= '\t', names= column_names)
moviesti= pd.read_csv('Movie_Id_Titles')

#merging the data and the movies titles in one data frame
data= pd.merge(data, moviesti, on= 'item_id')

#vizualisation of the data
data.groupby('title')['rating'].mean().sort_values(ascending= False).head()
data.groupby('title')['rating'].value_counts().sort_values(ascending= False).head()
moviesdata= pd.DataFrame(data.groupby('title')['rating'].mean())
moviesdata['no of rating']= pd.DataFrame(data.groupby('title')['rating'].count())

sns.jointplot(y= 'no of rating', x= 'rating', data= moviesdata)
sns.set_style('whitegrid')

moviesdata['rating'].plot(kind= 'hist', bins=70)

#creating the matrix type structure for the data
movie_mat= data.pivot_table(index='user_id', columns= 'title', values= 'rating')
moviesdata.sort_values('no of rating', ascending= False).head()

#Getting the rating given by the users to star war and liar liar movies
star_war_user= movie_mat['Star Wars (1977)']
liar_liar_user= movie_mat['Liar Liar (1997)']

#now finding the movies which stars war user like to watch
similar_starswar= movie_mat.corrwith(star_war_user)
#now finding the movies which liar liar user like to watch
similar_liarliar= movie_mat.corrwith(liar_liar_user)

similar_liarliar.dropna(inplace= True)
corr_liarliar= pd.DataFrame(similar_liarliar, columns= ['rating'])

similar_starswar.dropna(inplace= True)
corr_starwar= pd.DataFrame(similar_starswar, columns= ['rating'])

#top movies which most probabily stars war and liar liar viewer must like watch
corr_starwar.sort_values('rating', ascending= False).head()
corr_starwar= corr_starwar.join(moviesdata['no of rating'])
corr_starwar[corr_starwar['no of rating']>100].sort_values('rating', ascending= False).head()

corr_liarliar= corr_liarliar.join(moviesdata['no of rating'])
corr_liarliar[corr_liarliar['no of rating']>100].sort_values('rating', ascending= False).head()







