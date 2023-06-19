# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:55:06 2023

@author: Dell
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv(r'D:\Data Science\Daily Practice\March\14-03-2023\MLR\House_data.csv')
dataset.head()
dataset=dataset.drop(['id','date'],axis=1)

with sns.plotting_context("notebook",font_scale=2.5):
    g=sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']],
                   hue='bedrooms',palette='tab20',size=6)
    g.set(xticklabels=[]);

x=dataset.iloc[:,1:].values                  
y=dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)



from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_prd=regressor.predict(x_test)



import statsmodels.formula.api as sm

x=np.append(arr=np.ones((21613,1)).astype(int),values=x,axis=1)

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

