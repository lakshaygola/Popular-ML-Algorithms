#Finance Data Capstone Projrct 2
#In this project we import the data  online of the banks from the time of economic crisis 

from pandas_datareader import data , wb
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import plotly 
import cufflinks as cf
cf.go_offline()
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()

start = datetime.datetime(2006,1,1)
end = datetime.datetime(2016,1,1)

#Bank of america
BAC = data.DataReader('BAC', 'yahoo', start = start, end = end)
#CitiGroup
C = data.DataReader('C', 'yahoo', start = start, end = end)
#Goldman Sachs
GS = data.DataReader('GS', 'yahoo', start = start, end = end)
#JPMorgan Chase
JPM = data.DataReader('JPM', 'yahoo', start = start, end = end)
#Morgan Stanley
MS = data.DataReader('MS', 'yahoo', start = start, end = end)
#Wells Fargo
WFC = data.DataReader('WFC' , 'yahoo', start , end)

#list of ticker symbols
tickers = ['BAC' , 'C' , 'GS' , 'JPM' , 'MS' , 'WFC']

#concatenating all the dataframes
bank_stocks = pd.concat([BAC , C , GS , JPM , MS , WFC] , axis = 1 , keys = tickers)

#setting the columns names
bank_stocks.columns.names = ['Banks Tickers' , 'Stock Info']
bank_stocks.head()

#grouping by banks names
for tick in tickers:  
    print(tick , bank_stocks['BAC']['Close'].max())
#or
bank_stocks.xs(key = 'Close' , axis = 1 , level = 'Stock Info').max()

#Making new dataframe called return
returns = pd.DataFrame()

#calculating the percentage change on each rows in bank_stocks data
for tick in tickers:
    returns[tick + ' Return'] = bank_stocks[tick]['Close'].pct_change()

#pair plot 
sns.pairplot(data = returns[1:])
plt.tight_layout()

#best and worst dates for the particular banks in the return dataframe
returns.idxmin()

returns.idxmax()

#standard deviation of the return data frame
returns.std()

#standard deviation of the return data in 2015
returns.loc['2015-01-01':'2015-12-31'].std()

#Distplot of the 2015 returns for Morgan Stanley
sns.distplot(returns.loc['2015-01-01' : '2015-12-31']['MS Return'] ,bins = 30 , color='green')
sns.set_style('whitegrid')

#2008 citigroup distplot
sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'] , bins = 50 , color = 'Red')

#line plot for each bank 
#Using for loop
for tick in tickers:
    bank_stocks[tick]['Close'].plot(label = tick)
plt.legend()

#line plot for each bank
#Using .xs method
bank_stocks.xs(key ="Close" , level = 'Stock Info'  , axis = 1).plot()

#Using plotly
bank_stocks.xs(key = 'Close' , level = 'Stock Info' , axis = 1).iplot()

#Ploting the rolling average of BAC for the year 2008
bank_stocks['BAC']['Close'].loc['2008-01-01':'2009-01-01'].rolling(window = 30).mean().plot()
bank_stocks['BAC']['Close'].loc['2008-01-01':'2009-01-01'].plot()

#Heat map of the close columns 
close_corr = bank_stocks.xs(key = 'Close' , axis = 1 , level = 'Stock Info').corr()
sns.heatmap(close_corr,annot = True)

#Cluster map
sns.clustermap(close_corr , annot = True)

#Heat map using iplot
close_corr.iplot(kind = 'heatmap')

#Candle plot of bank of america from 2015 to 2016
bank_stocks['BAC'][['Open','High','Low','Close']].loc['2015-01-01':'2016-01-01'].iplot(kind = 'candle')

#Simple moving averages plot of the morgan stanley for the year 2015
bank_stocks['MS'].loc['2015-01-01':'2015-12-31'].ta_plot(study = 'sma')

#Bollinger band plot for the Bank of america for the year 2015
bank_stocks['BAC'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll')








