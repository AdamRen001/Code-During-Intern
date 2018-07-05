
# coding: utf-8

# In[1]:

import re
import os
import numpy as np
import pandas as pd
from sklearn import linear_model


# In[2]:
def get_date_map(dir_, date_start, date_end, minsize=1e6):
    date_start, date_end = map(str, [date_start, date_end])
    files = sorted(os.listdir(dir_))
    rval = {}
    for f in files:
        if f[-3:]=='csv':
         match =  re.search('(\d{8})', f)
         if match is not None:
            date = match.group(1)
            if os.path.getsize(os.path.join(dir_, f)) >= minsize  and date >= date_start and date <= date_end:
                rval[date] = os.path.join(dir_, f)
    return rval




#ols regression
from sklearn.linear_model import LinearRegression
def ols_regression(X,y,X_test):
    linreg = LinearRegression(n_jobs=-1)
    linreg.fit(X,y)
    y_pred = linreg.predict(X_test)
    pred_ols = pd.DataFrame(y_pred)
    return pred_ols


# In[3]:


#ridge regression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
def ridge_regression(X,y,X_test):
    ridge_reg = RidgeCV(alphas=[0.05,0.1,0.5,1.0,2,5])
    ridge_reg.fit(X,y)
    y_rid_pred = ridge_reg.predict(X_test)
    best_alpha = ridge_reg.alpha_
    print ("Best Alpha for Ridge:", best_alpha)
    pred_ridge = pd.DataFrame(y_rid_pred)
    return pred_ridge,best_alpha

def ridge(X,y,X_test,best_alpha):
    ridge_reg = Ridge(alpha=best_alpha)
    ridge_reg.fit(X,y)
    y_rid_pred = ridge_reg.predict(X_test)
    pred_ridge = pd.DataFrame(y_rid_pred)
    return pred_ridge
# In[4]:


#lasso regression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
def lasso_regression(X,y,X_test):
    lasso_reg = LassoCV(alphas=[0.05,0.1,0.5,1],tol=0.001,n_jobs=-1)
    lasso_reg.fit(X,y)
    y_lasso_pred = lasso_reg.predict(X_test)
    pred_lasso = pd.DataFrame(y_lasso_pred)
    best_alpha = lasso_reg.alpha_
    print("Best Alpha for Lasso:", best_alpha)
    return pred_lasso,best_alpha

def lasso(X,y,X_test,best_alpha):
    lasso_reg = Lasso(alpha=best_alpha,tol=0.001)
    lasso_reg.fit(X,y)
    y_lasso_pred = lasso_reg.predict(X_test)
    pred_lasso = pd.DataFrame(y_lasso_pred)
    return pred_lasso

# In[5]:


#elasticnet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
def elastic_regression(X,y,X_test):
    elastic_net = ElasticNetCV(alphas=[0,0.5,1.0,5.0,10],
                               l1_ratio=[.1, .5, .9, 0.95,0.99],
                               tol=0.001, max_iter=5000,n_jobs=-1)
    elastic_net.fit(X, y)
    y_elastic_pred = elastic_net.predict(X_test)
    pred_elastic = pd.DataFrame(y_elastic_pred)
    best_alpha = elastic_net.alpha_
    best_l1_ratio = elastic_net.l1_ratio_
    print ("Best Alpha for ElasticNet:", best_alpha)
    print ("L1 Ratio for ElasticNet:", best_l1_ratio)
    return pred_elastic,best_alpha,best_l1_ratio

def elastic(X,y,X_test,best_alpha,best_l1_ratio):
    elastic_net = ElasticNet(alpha=best_alpha,
                             l1_ratio=best_l1_ratio,tol=0.001)
    elastic_net.fit(X, y)
    y_elastic_pred = elastic_net.predict(X_test)
    pred_elastic = pd.DataFrame(y_elastic_pred)
    return pred_elastic

#GradientBoosting
from sklearn.ensemble import GradientBoostingRegressor
def Gradient_Boosting(X,y,X_test):
    GB = GradientBoostingRegressor()
    GB.fit(X, y)
    y_GB_pred = GB.predict(X_test)
    pred_GB = pd.DataFrame(y_GB_pred)
    return pred_GB
# In[6]:

startdate = '20180104'
enddate = '20180201'
file_paths = get_date_map('/home/tysongroup/lowdim/',startdate,enddate)
file_paths = sorted(file_paths.values())
da = pd.read_csv(file_paths[3],header=None)
tradingday = da.iloc[:,0]
updatetime = da.iloc[:,1]
y_test = da.iloc[:,len(da.columns)-1]
for f in file_paths[4:len(file_paths)]:
    df = pd.read_csv(f,header=None)
    tradingday = pd.concat([tradingday,df.iloc[:,0]])
    updatetime = pd.concat([updatetime,df.iloc[:,1]])
    y_test = pd.concat([y_test,df.iloc[:,len(df.columns)-1]])
y_test = abs(y_test)


data1 = pd.read_csv(file_paths[0],header=None)
for f in file_paths[1:3]:
    df = pd.read_csv(f,header=None)
    data1 = pd.concat([data1,df])
y1 = abs(data1.iloc[:,len(data1.columns)-1])
X1 = abs(data1.iloc[:,2:len(data1.columns)-1])
df1 = pd.read_csv(file_paths[3],header=None)
X_test = abs(df1.iloc[:,2:len(df1.columns)-1])
pred_ols = ols_regression(X1,y1,X_test)
pred_ridge,best_alpha1 = ridge_regression(X1,y1,X_test)
pred_lasso,best_alpha2 = lasso_regression(X1,y1,X_test)
pred_elastic,best_alpha3,best_l1_ratio = elastic_regression(X1,y1,X_test)
pred_GB = Gradient_Boosting(X1,y1,X_test)


# In[ ]:


for i in range(1,len(file_paths)-3):
    data = pd.read_csv(file_paths[i],header=None)
    for f in file_paths[i+1:i+3]:
        df = pd.read_csv(f,header=None)
        data = pd.concat([data,df])
    y = abs(data.iloc[:,len(data.columns)-1])
    X = abs(data.iloc[:,2:len(data.columns)-1])
    df = pd.read_csv(file_paths[i+3],header=None)
    X_test = abs(df.iloc[:,2:len(df.columns)-1])
    pred_ols = pd.concat([pred_ols,
                          ols_regression(X,y,X_test)])
    pred_ridge = pd.concat([pred_ridge,
                            ridge(X,y,X_test,best_alpha1)])
    pred_lasso = pd.concat([pred_lasso,
                            lasso(X,y,X_test,best_alpha2)])
    pred_elastic = pd.concat([pred_elastic,
                              elastic(X,y,X_test,best_alpha3,best_l1_ratio)])
    pred_GB = pd.concat([pred_GB,Gradient_Boosting(X,y,X_test)])

# In[ ]:


from sklearn.metrics import r2_score
r2_ols = r2_score(y_test,pred_ols)
r2_ridge = r2_score(y_test,pred_ridge)
r2_lasso = r2_score(y_test,pred_lasso)
r2_elastic = r2_score(y_test,pred_elastic)
r2_GBR = r2_score(y_test,pred_GB)
r_squared= pd.DataFrame([str(r2_ols*100)+'%',str(r2_ridge*100)+'%',
                         str(r2_lasso*100)+'%',str(r2_elastic*100)+'%',
                         str(r2_GBR*100)+'%'],
                         columns=['20180109-20180201 R^2'],
                         index=['OLS','Ridge','Lasso','ElasticNet','GBR'])


# In[ ]:


print(r2_ols)
print(r2_ridge)
print(r2_lasso)
print(r2_elastic)
print(r2_GBR)
r_squared.to_csv('/home/rzg/result.csv')

method1 = pd.DataFrame({'TradingDay':tradingday,
                        'UpdateTime':updatetime,
                        'y_predict':pred_ols.iloc[:,0]
                       })
method1.to_csv('/home/rzg/project1/opredict_ols.csv',index=False)

method2 = pd.DataFrame({'TradingDay':tradingday,
                        'UpdateTime':updatetime,
                        'y_predict':pred_ridge.iloc[:,0]
                       })
method2.to_csv('/home/rzg/project1/ypredict_ridge.csv',index=False)

method3 = pd.DataFrame({'TradingDay':tradingday,
                        'UpdateTime':updatetime,
                        'y_predict':pred_lasso.iloc[:,0]
                       })
method3.to_csv('/home/rzg/project1/ypredict_lasso.csv',index=False)

method4 = pd.DataFrame({'TradingDay':tradingday,
                        'UpdateTime':updatetime,
                        'y_predict':pred_elastic.iloc[:,0]
                       })
method4.to_csv('/home/rzg/project1/ypredict_elastic.csv',index=False)

method5 = pd.DataFrame({'TradingDay':tradingday,
                        'UpdateTime':updatetime,
                        'y_predict':pred_GB.iloc[:,0]
                       })
method5.to_csv('/home/rzg/project1/ypredict_GBR.csv',index=False)
