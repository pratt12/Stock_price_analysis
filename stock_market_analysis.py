#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Stock Market project (monte-carlo technique)
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

#Visualization plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# For reading stock data from yahoo
import yfinance as yf
from datetime import datetime

# For division
from __future__ import division



# In[4]:


from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()

spy = pdr.get_data_yahoo('SPY', start='2022-10-24', end='2022-12-23')

spy


# In[9]:


# The tech stocks we'll use for this analysis
tech_list = ['AAPL','GOOG','MSFT','AMZN']
from pandas_datareader import data as pdr
# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1,end.month,end.day)

# Download stock data for each symbol
for stock in tech_list:
    globals()[stock] = pdr.get_data_yahoo(stock, start, end)


# In[10]:


AAPL


# In[11]:


GOOG.head()


# In[13]:


AAPL.describe()


# In[14]:


AAPL.info()


# In[15]:


AAPL['Adj Close'].plot(legend=True, figsize=(10,4))


# In[17]:


AAPL['Volume'].plot(legend=True, figsize=(10,4))
#daily traded volume of the stocks for the past year


# In[18]:


#Now we look for the moving average


# In[19]:


#We need to plot moving averages for these stock proces, Pandas has inbuilt feature for this

ma_day=[10,20,50]

for ma in ma_day:
    column_name="MA for %s days" %(str(ma))
    
    AAPL[column_name] = AAPL['Adj Close'].rolling(window=ma).mean()


# In[20]:


AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(subplots=False, figsize=(10,4))


# In[21]:


#Analysis of daily returns and risk of the stock

AAPL['Daily Return']= AAPL['Adj Close'].pct_change()

#plotting the return
AAPL['Daily Return'].plot(figsize=(10,4),legend=True, linestyle='--', marker='o')


# In[22]:


sns.distplot(AAPL['Daily Return'].dropna(),bins=100, color='green')


# In[23]:


MSFT.head()

#from pandas_datareader import data as pdr
#data = pdr.DataReader(stock, 'yahoo', start, end)
#print(data)
#closing_df = pdr.DataReader(tech_list,'yahoo',start,end)['Adj Close']


# In[25]:


AAPL['Daily Return'].hist(bins=100)


# In[30]:


closing_df = pdr.get_data_yahoo(tech_list,start,end)['Adj Close']


# In[31]:


closing_df


# In[32]:


tech_rets = closing_df.pct_change()


# In[33]:


tech_rets.head()


# In[35]:


sns.jointplot('GOOG','GOOG',tech_rets, kind='scatter', color='seagreen')


# In[36]:


sns.jointplot('GOOG','MSFT', tech_rets, color='blue')


# In[37]:


#pearson value or pearson product-moment correlation coeff
#describes the correlation bw the values


# In[38]:


sns.pairplot(tech_rets.dropna())


# In[41]:


#controlling the correlation fig using seaborn

returns_fig=sns.PairGrid(tech_rets.dropna())

returns_fig.map_upper(plt.scatter,color='purple')

returns_fig.map_lower(sns.kdeplot, cmap='cool_d')

returns_fig.map_diag(plt.hist, bins=30)


# In[42]:


#controlling the correlation fig using seaborn

returns_fig=sns.PairGrid(closing_df)

returns_fig.map_upper(plt.scatter,color='purple')

returns_fig.map_lower(sns.kdeplot, cmap='cool_d')

returns_fig.map_diag(plt.hist, bins=30)


# In[48]:


# Compute the correlation matrix
corr_matrix = tech_rets.dropna().corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Tech Stock Returns')
plt.show()


# In[53]:


corr_matrix = closing_df.dropna().corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Tech Stock Closing Price')
plt.show()


# In[54]:


#Risk Analysis of the stocks 
rets = tech_rets.dropna()


# In[62]:


area=np.pi*20

plt.scatter(rets.mean(),rets.std(),s=area)

plt.xlabel('Expected Return')
plt.ylabel('Risk') 

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
    label,
    xy=(x,y),xytext=(50,50),
    textcoords= 'offset points', ha='right', va='bottom',
    arrowprops = dict(arrowstyle='-', connectionstyle='arc3, rad=-0.6'))


# In[63]:


sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color='purple')


# In[64]:


#0.05 quantile means 95% confidence of the worst % loss per day 
#using montecarlo method.
rets['AAPL'].quantile(0.05)


# In[75]:


# i.e. a 1.8% loss at the worst per day 

days = 365

dt=1/days

mu= rets.mean()['GOOG'] #expected return

sigma = rets.std()['GOOG'] #stock volatility


# In[79]:


def stock_monte_carlo(start_pr, days, mu, sigma):
    price=np.zeros(days)
    price[0]=start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1,days):
        shock[x] = np.random.normal(loc=mu*dt, scale= sigma*np.sqrt(dt))
        
        drift[x]=mu*dt
        
        price[x]=price[x-1]+ (price[x-1]*drift[x]*shock[x])
        
    return price


# In[80]:


GOOG.head()


# In[81]:


start_price=101.71
mu= rets.mean()['GOOG'] #expected return

sigma = rets.std()['GOOG'] #stock volatility

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))

plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')


# In[83]:


runs=1000

simulation = np.zeros(runs)

for run in range(runs):
    simulation[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]


# In[84]:


q=np.percentile(simulation,1)

plt.hist(simulation, bins=200)

plt.figtext(0.6,0.8, "VaR(0.99): $%.2f" % q)

plt.axvline(x=q, linewidth=4, color='r')

plt.title(u"Final price distribution for Google Stock after %days" %days, weight='bold');


# In[ ]:




