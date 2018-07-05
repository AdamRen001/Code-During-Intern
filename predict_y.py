
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
    linreg = LinearRegression(fit_intercept=False,n_jobs=-1)
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
    lasso_reg = LassoCV(alphas=[0.05,0.1,0.5,1],tol=0.001)
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
    elastic_net = ElasticNetCV(alphas=[0.5,1.0,5.0,10],
                               l1_ratio=[.1, .5, .9, 0.95,0.99],
                               tol=0.001, max_iter=5000)
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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
def GBRCV(X,y,X_test):
    estimator = GradientBoostingRegressor()
    param_grid = {
	          'loss':['ls','huber'],
	          'n_estimators':[100,200,300],
	          'learning_rate': [0.05,0.5,1],
                  'max_depth': [3, 5,10],
		}
    GB = GridSearchCV(estimator,param_grid)
    GB.fit(X,y)
    y_GB_pred = GB.predict(X_test)
    pred_GB = pd.DataFrame(y_GB_pred)
    print GB.best_estimator_
    print GB.best_params_
    return pred_GB,GB.best_params_
def Gradient_Boosting(X,y,X_test):
    GB = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                   learning_rate=0.05, loss='huber', max_depth=5,
                                   max_features=None, max_leaf_nodes=None,
                                   min_impurity_split=1e-07, min_samples_leaf=1,
                                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                                   n_estimators=50, presort='auto', random_state=None,
                                   subsample=1.0, verbose=0, warm_start=False)

    GB.fit(X, y)
    y_GB_pred = GB.predict(X_test)
    pred_GB = pd.DataFrame(y_GB_pred)
    return pred_GB
# In[6]:
y_sign = pd.read_csv('/home/rzg/ypredict_lgbm.csv')
print "len y_sign=",len(y_sign)
startdate = '20180109'
enddate = '20180201'
file_paths = get_date_map('/home/tysongroup/lowdim/',startdate,enddate)
file_paths = sorted(file_paths.values())
d1 = pd.read_csv(file_paths[0],header=None)
d2 = pd.read_csv(file_paths[1],header=None)
d3 = pd.read_csv(file_paths[2],header=None)
d4 = pd.read_csv(file_paths[3],header=None)
d5 = pd.read_csv(file_paths[4],header=None)
d6 = pd.read_csv(file_paths[5],header=None)
index = len(d1) + len(d2) + len(d3)+len(d4)+len(d5)+len(d6)
print ('index=',index)
da = pd.read_csv(file_paths[6],header=None)
for f in file_paths[7:len(file_paths)]:
    df = pd.read_csv(f,header=None)
    da = pd.concat([da,df])
da.insert(len(da.columns),'y_sign',y_sign.iloc[index:,2].values)
da_pos = da[da['y_sign']>0]
da_neg = da[da['y_sign']<0]
tradingday_pos = da_pos.iloc[:,0]
tradingday_neg = da_neg.iloc[:,0]
updatetime_pos = da_pos.iloc[:,1]
updatetime_neg = da_neg.iloc[:,1]
y_positive = da_pos.iloc[:,len(da_pos.columns)-2]
y_negative = da_neg.iloc[:,len(da_neg.columns)-2]
y_test = pd.concat([y_positive,y_negative])
print ('len y_test=',len(y_test))
tradingday = pd.concat([tradingday_pos,tradingday_neg])
print ('len day=',len(tradingday))
updatetime = pd.concat([updatetime_pos,updatetime_neg])
print ('len time=',len(updatetime))

index = len(d1) + len(d2) + len(d3)+len(d4)+len(d5)+len(d6)
data1 = pd.read_csv(file_paths[0],header=None)
for f in file_paths[1:6]:
    df = pd.read_csv(f,header=None)
    data1 = pd.concat([data1,df])
data1.insert(len(data1.columns),'y_sign',y_sign.iloc[:index,2].values)	
data1_pos = data1[data1['y_sign']>0]
data1_neg = data1[data1['y_sign']<0]
y1_pos = data1_pos.iloc[:,len(data1_pos.columns)-2]
y1_neg = data1_neg.iloc[:,len(data1_neg.columns)-2]
X1_pos = data1_pos.iloc[:,2:len(data1_pos.columns)-2]
X1_neg = data1_neg.iloc[:,2:len(data1_neg.columns)-2]
print data1_pos.head(3)
print y1_pos.head()
print X1_pos.head(3)

df1 = pd.read_csv(file_paths[6],header=None)
df1.insert(len(df1.columns),'y_sign',y_sign.iloc[index:index+len(df1),2].values)
df1_pos = df1[df1['y_sign']>0]
df1_neg = df1[df1['y_sign']<0]
print df1_pos.head(3)
X_test_pos = df1_pos.iloc[:,2:len(df1_pos.columns)-2]
print '+x:',X_test_pos.head(3)
X_test_neg = df1_neg.iloc[:,2:len(df1_neg.columns)-2]
pred_ols_pos = ols_regression(X1_pos,y1_pos,X_test_pos)
pred_ols_neg = ols_regression(X1_neg,y1_neg,X_test_neg)
pred_ridge_pos,best_alpha1_pos = ridge_regression(X1_pos,y1_pos,X_test_pos)
pred_ridge_neg,best_alpha1_neg = ridge_regression(X1_neg,y1_neg,X_test_neg)
pred_lasso_pos,best_alpha2_pos = lasso_regression(X1_pos,y1_pos,X_test_pos)
pred_lasso_neg,best_alpha2_neg = lasso_regression(X1_neg,y1_neg,X_test_neg)
pred_elastic_pos,best_alpha3_pos,best_l1_ratio_pos = elastic_regression(X1_pos,y1_pos,X_test_pos)
pred_elastic_neg,best_alpha3_neg,best_l1_ratio_neg = elastic_regression(X1_neg,y1_neg,X_test_neg)
pred_GB_pos = Gradient_Boosting(X1_pos,y1_pos,X_test_pos)
pred_GB_neg = Gradient_Boosting(X1_neg,y1_neg,X_test_neg)

# In[ ]:

index = len(d1)
for i in range(1,len(file_paths)-6):
    data = pd.read_csv(file_paths[i],header=None)
    print("index=",index)
    increase = len(data)
    index_2 = len(data)
    for f in file_paths[i+1:i+6]:
        df = pd.read_csv(f,header=None)
        data = pd.concat([data,df])
	index_2 = index_2 + len(df) 
    print ("index_2=",index_2)
    print (index+index_2)
    data.insert(len(data.columns),'y_sign',y_sign.iloc[index:index+index_2,2].values)
      
    data_pos = data[data['y_sign']>0]
    data_neg = data[data['y_sign']<0]
    y_pos = data_pos.iloc[:,len(data_pos.columns)-2]
    y_neg = data_neg.iloc[:,len(data_neg.columns)-2]
    X_pos = data_pos.iloc[:,2:len(data_pos.columns)-2]
    X_neg = data_neg.iloc[:,2:len(data_neg.columns)-2]
    df = pd.read_csv(file_paths[i+6],header=None)
    df.insert(len(df.columns),'y_sign',y_sign.iloc[index+index_2:index+index_2+len(df),2].values)
    df_pos = df[df['y_sign']>0]
    df_neg = df[df['y_sign']<0]
    X_test_pos = df_pos.iloc[:,2:len(df_pos.columns)-2]
    X_test_neg = df_neg.iloc[:,2:len(df_neg.columns)-2]
    index = index + increase
    pred_ols_pos = pd.concat([pred_ols_pos,
	                      ols_regression(X_pos,y_pos,X_test_pos)])
    pred_ols_neg = pd.concat([pred_ols_neg,
	                      ols_regression(X_neg,y_neg,X_test_neg)])
    pred_ridge_pos = pd.concat([pred_ridge_pos,
                                ridge(X_pos,y_pos,X_test_pos,best_alpha1_pos)])
    pred_ridge_neg = pd.concat([pred_ridge_neg,
                                ridge(X_neg,y_neg,X_test_neg,best_alpha1_neg)])						
    pred_lasso_pos = pd.concat([pred_lasso_pos,
                                lasso(X_pos,y_pos,X_test_pos,best_alpha2_pos)])
    pred_lasso_neg = pd.concat([pred_lasso_neg,
                                lasso(X_neg,y_neg,X_test_neg,best_alpha2_neg)])
    pred_elastic_pos = pd.concat([pred_elastic_pos,
                                  elastic(X_pos,y_pos,X_test_pos,best_alpha3_pos,best_l1_ratio_pos)])
    pred_elastic_neg = pd.concat([pred_elastic_neg,
                                  elastic(X_neg,y_neg,X_test_neg,best_alpha3_neg,best_l1_ratio_neg)])
    pred_GB_pos = pd.concat([pred_GB_pos,Gradient_Boosting(X_pos,y_pos,X_test_pos)])
    pred_GB_neg = pd.concat([pred_GB_neg,Gradient_Boosting(X_neg,y_neg,X_test_neg)])
#In[ ]:
pred_ols = pd.concat([pred_ols_pos,pred_ols_neg])
pred_ridge = pd.concat([pred_ridge_pos,pred_ridge_neg])
pred_lasso = pd.concat([pred_lasso_pos,pred_lasso_neg])
pred_elastic = pd.concat([pred_elastic_pos,pred_elastic_neg])
pred_GB = pd.concat([pred_GB_pos,pred_GB_neg])
print('len ols=',len(pred_ols))
print('len ridge=',len(pred_ridge))
print('len lasso=',len(pred_lasso))
print('len elasticnet=',len(pred_elastic))
print('len GBR=',len(pred_GB))
from sklearn.metrics import r2_score
r2_ols = r2_score(y_test,pred_ols)
print ('R^2 OLS=',r2_ols)
r2_ridge = r2_score(y_test,pred_ridge)
print ('R^2 ridge=',r2_ridge)
r2_lasso = r2_score(y_test,pred_lasso)
print ('R^2 lasso=',r2_lasso)
r2_elastic = r2_score(y_test,pred_elastic)
print ('R^2 elastic=',r2_elastic)
r2_GBR = r2_score(y_test,pred_GB)
print ('R^2 gbr=',r2_GBR)
r_squared= pd.DataFrame([str(r2_ols*100)+'%',str(r2_ridge*100)+'%',
                         str(r2_lasso*100)+'%',str(r2_elastic*100)+'%'],
                         columns=['20180112-20180201 R^2'],
                         index=['OLS','Ridge','Lasso','ElasticNet'])

# In[ ]:
r_squared.to_csv('/home/rzg/result.csv')

method1 = pd.DataFrame({'TradingDay':tradingday.values,
                        'UpdateTime':updatetime.values,
                        'y_predict':pred_ols.iloc[:,0]
                       })
method1.sort_values(by=['TradingDay','UpdateTime'],inplace=True)
method1.to_csv('/home/rzg/ypredict_ols.csv',index=False)

method2 = pd.DataFrame({'TradingDay':tradingday.values,
                        'UpdateTime':updatetime.values,
                        'y_predict':pred_ridge.iloc[:,0]
                       })
method2.sort_values(by=['TradingDay','UpdateTime'],inplace=True)
method2.to_csv('/home/rzg/ypredict_ridge.csv',index=False)

method3 = pd.DataFrame({'TradingDay':tradingday.values,
                        'UpdateTime':updatetime.values,
                        'y_predict':pred_lasso.iloc[:,0]
                       })
method3.sort_values(by=['TradingDay','UpdateTime'],inplace=True)
method3.to_csv('/home/rzg/ypredict_lasso.csv',index=False)

method4 = pd.DataFrame({'TradingDay':tradingday.values,
                        'UpdateTime':updatetime.values,
                        'y_predict':pred_elastic.iloc[:,0]
                       })
method4.sort_values(by=['TradingDay','UpdateTime'],inplace=True)
method4.to_csv('/home/rzg/ypredict_elastic.csv',index=False)

method5 = pd.DataFrame({'TradingDay':tradingday.values,
                        'UpdateTime':updatetime.values,
                        'y_predict':pred_GB.iloc[:,0]
                       })
method5.sort_values(by=['TradingDay','UpdateTime'],inplace=True)
method5.to_csv('/home/rzg/ypredict_GBR.csv',index=False)
