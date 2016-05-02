import pandas as pd
from collections import defaultdict
import datetime
from matplotlib import pyplot as plt
# Make the plots bigger
plt.rcParams['figure.figsize'] = 10, 10
import seaborn as sns
import numpy as np
from sklearn import linear_model
% matplotlib inline

df = pd.read_csv('Iowa_Liquor_Sales_reduced.csv')


#Cleaning the Data:
df['Date']=pd.to_datetime(df['Date'], infer_datetime_format=True)


#Drop the Dollar sign from columns and convert the remaining values to floats

df['State Bottle Cost'].replace(to_replace='\$', value='', inplace=True, regex=True)
df['State Bottle Cost']=df['State Bottle Cost'].astype(float)


df['State Bottle Retail'].replace(to_replace='\$', value='', inplace=True, regex=True)
df['State Bottle Retail']=df['State Bottle Retail'].astype(float)



df['Sale (Dollars)'].replace(to_replace='\$', value='', inplace=True, regex=True)
df['Sale (Dollars)']=df['Sale (Dollars)'].astype(float)


pd.options.display.float_format = '{:.2f}'.format
df.describe()
df.ix[[df['Bottle Volume (ml)'].argmax()]]
df['Category Name'].nunique()

# What type of data should the store number be stored as?
df['Store Number'].nunique()

# There are two sales that have a bottle volume of 0 ml. Unless this is ghost
# vodka, there was an error somewhere. After doing research, I'd like to replace them
# with what probably should be there. Based on research, that should be 100ml.
df[df['Bottle Volume (ml)']==0]
df[df['Bottle Volume (ml)']==0] = 100

#Rename the columns to make them easier to work with from here on out
df.rename(columns={'Store Number': 'store', 'Zip Code':'zip', 'County Number':'county',
        'Category Name':'category', 'Vendor Number':'vendor', 'Item Number':'item',
        'Bottle Volume (ml)':'mls','State Bottle Cost':'state_cost', 'State Bottle Retail':'state_retail',
        'Bottles Sold':'bottles_sold','Sale (Dollars)':'sale', 'Volume Sold (Liters)':'liters_sold'}, inplace=True)
#There isn't a column that had the toal sale in it. There's a column for 'Sale (Dollars)'
# that isn't especially useful. It should be renamed to be simpler, and then its values should be replaced
# with

# ##### Testing Zone. This smaller data frame is where I'll test out the functions I'm running
# ##### on the larger one to make sure they don't do anything wonky before running code.
#
# testdf= df.ix[0:50, :]
# testdf.columns
#


#Create a column for margins
df['margin'] = df['state_retail'] - df['state_cost']


#Create a price per liter column

df['ppl']=(df['state_retail']/df['mls'])*1000

df.columns
df.ppl.head()

df.pivot_table(index=['store', 'Date'], columns='category', values='sale', margins=True, aggfunc=np.sum)

# Create a mask that only contains the 2015 sales. Building a model off this will allow us to model 2016 sales.
start_date = pd.Timestamp('20150101')
end_date = pd.Timestamp('20151231')
mask = (df['Date'] >= start_date) & (df['Date']<=end_date)
sales=df[mask]
sales.head()
import pandas as pd

# This little number gives you the total for each store's sales.
sales.groupby('store').sale.sum()
sales.groupby('store').category.value_counts()

stores = sales.groupby(by=['store'], as_index=False)

sales.sale.sum()

sales = sales.agg({"sale": [np.sum, np.mean, np.count_nonzero],
                   "liters_sold": [np.sum, np.mean],
                   "margin": np.mean,
                   "ppl": np.mean,
                   "zip": lambda x: x.iloc[0], # just extract once, should be the same
                   "City": lambda x: x.iloc[0],
                   "county": lambda x: x.iloc[0],
                   "Date": (np.min, np.max)})


sales.columns = ['_'.join(col).strip() for col in sales.columns.values]
sales.columns

sales.sale_mean.describe()
df.sale.describe()
sales.tail()
#Start making some graphs to pick out correlations.

# There's a super long tail on the sales data. Some of the stores have a really high
# average sale.
sns.distplot(df[df.sale <= 200].sale)
df.dtypes
sns.tsplot(time='Date', value='sale', data=df)

sales_fig =sns.pairplot(x_vars=['sale_mean', 'sale_sum', 'sale_count_nonzero', 'ppl_mean', 'margin_mean', 'liters_sold_mean'],
                        y_vars=['sale_mean', 'sale_sum', 'sale_count_nonzero', 'ppl_mean', 'margin_mean', 'liters_sold_mean'],
                        data = sales)

sales_fig.savefig('sales_fig.png')
