
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
# In[41]:
def Lin_regression(X,y,X_test):
    lin_reg = LinearRegression(n_jobs=10,fit_intercept=False)
    lin_reg.fit(X,y)
    y_lin_pred = lin_reg.predict(X_test)
    pred_lin = pd.DataFrame(y_lin_pred)
    y_lin_test = lin_reg.predict(X)
    test_lin = pd.DataFrame(y_lin_test)
    return pred_lin,test_lin


def Gradient_Boosting(X,y,X_test):
    GB = GradientBoostingRegressor(n_estimators=10,loss='ls',warm_start=True)
    GB.fit(X, y)
    y_GB_pred = GB.predict(X_test)
    pred_GB = pd.DataFrame(y_GB_pred)
    y_GB_test = GB.predict(X)
    test_GB = pd.DataFrame(y_GB_test)
    return pred_GB,test_GB


# In[42]:
def ridge_regression(X,y,X_test):
    ridge_reg = Ridge(alpha=0.05)
    ridge_reg.fit(X,y)
    y_ridge_pred = ridge_reg.predict(X_test)
    pred_ridge = pd.DataFrame(y_ridge_pred)
    y_ridge_test = ridge_reg.predict(X)
    test_ridge = pd.DataFrame(y_ridge_test)
    return pred_ridge,test_ridge


def lasso_regression(X,y,X_test):
    lasso_reg = Lasso(alpha=0.001,tol=0.0001)
    lasso_reg.fit(X,y)
    y_lasso_pred = lasso_reg.predict(X_test)
    pred_lasso = pd.DataFrame(y_lasso_pred)
    y_lasso_test = lasso_reg.predict(X)
    test_lasso = pd.DataFrame(y_lasso_test)
    return pred_lasso,test_lasso


# In[52]:


def Bagging_reg(X,y,X_test):
    BR = BaggingRegressor(SVR(kernel='linear',epsilon=0.01),n_estimators=5,warm_start=True)
    #BR = BaggingRegressor(base_estimator=LinearRegression(fit_intercept=False),n_estimators=5)
    BR.fit(X, y)
    y_BR_pred = BR.predict(X_test)
    pred_BR = pd.DataFrame(y_BR_pred)
    y_BR_test = BR.predict(X)
    test_BR = pd.DataFrame(y_BR_test)
    return pred_BR,test_BR

def RF_reg(X,y,X_test):
    RF = RandomForestRegressor(max_depth=2,n_jobs=10,n_estimators=8)
    RF.fit(X, y)
    y_RF_pred = RF.predict(X_test)
    pred_RF = pd.DataFrame(y_RF_pred)
    y_RF_test = RF.predict(X)
    test_RF = pd.DataFrame(y_RF_test)
    return pred_RF,test_RF


def LGBM(X,y,X_test):
    LGBM_reg =  LGBMRegressor(n_jobs=10,num_leaves=15,n_estimators=20,subsample_for_bin=150)
    LGBM_reg.fit(X,y)
    y_LGBM_reg = LGBM_reg.predict(X_test)
    pred_LGBM = pd.DataFrame(y_LGBM_reg)
    y_LGBM_test = LGBM_reg.predict(X)
    test_LGBM = pd.DataFrame(y_LGBM_test)
    return pred_LGBM,test_LGBM

# In[44]:


def standard(x):
    return (x-x.mean())/(x.std())


# In[45]:


correlation = pd.read_csv('/home/rzg/corrdata.csv',header=None)
names = pd.read_csv('/home/rzg/names.csv',header=None)
names = np.array(names).ravel()
correlation = correlation.T
correlation.columns = names
selectcols = list(correlation.columns[abs(correlation.iloc[2,:]).argsort()[-50:]])
print selectcols

# In[46]:


alldata = pd.read_csv('/home/rzg/alldata.csv')

# In[48]:
matrix = alldata[selectcols+['date']+['D00000001']]
matrix.dropna(inplace=True)
X_all = matrix[matrix.columns[0:len(matrix.columns)-2]]
y_all = matrix['D00000001']
date = matrix['date']
#X_all = pd.ewma(X_all,span=10)
# In[49]:


# In[50]:


y_true = pd.DataFrame()
y_predict_ridge = pd.DataFrame()
y_predict_lin = pd.DataFrame()
y_predict_lasso = pd.DataFrame()
y_predict_GB = pd.DataFrame()
y_predict_RF = pd.DataFrame()
y_predict_BR = pd.DataFrame()
y_predict_LGBM = pd.DataFrame()
day = pd.DataFrame()
R_lin = []
R_ridge = []
R_lasso = []
R_BR = []
R_GB = []
R_RF = []
R_LGBM = []
for i in range(0,len(X_all)-751):
    X_train = X_all[i:i+750]
    y_train = y_all[i:i+750]
    new_matrix = pd.concat([X_train,y_train],axis=1)
    print new_matrix.corr().head()
    sel = list(abs(new_matrix.corr()['D00000001']).argsort()[-4:-1].index)
    print sel
    X_train = X_train[sel]
    X_test = X_all[i+750:i+751]
    X_test = X_test[sel]
    y_test = y_all[i+750:i+751]
    test_date = date[i+750:i+751]
    predict_lin,test_lin = Lin_regression(standard(X_train),y_train,(X_test-X_train.mean())/(X_train.std()))

    y_predict_lin = pd.concat([y_predict_lin,predict_lin])

    y_train_1 = y_train.reshape(-1,1)
    SST = ((y_train_1-y_train_1.mean())**2).sum()
    SSE_1 = ((np.array(y_train_1)-np.array(test_lin)) ** 2).sum() 
    R_lin.append(1-SSE_1/SST) 
   
    predict_ridge,test_ridge = ridge_regression(standard(X_train),y_train,(X_test-X_train.mean())/(X_train.std()))
    y_predict_ridge = pd.concat([y_predict_ridge,predict_ridge])
    SSE_2 = ((np.array(y_train_1)-np.array(test_ridge)) ** 2).sum()
    R_ridge.append(1-SSE_2/SST)
    
    predict_lasso,test_lasso = lasso_regression(standard(X_train),y_train,(X_test-X_train.mean())/(X_train.std()))
    y_predict_lasso = pd.concat([y_predict_lasso,predict_lasso])
    SSE_3 = ((np.array(y_train_1)-np.array(test_lasso)) ** 2).sum()
    R_lasso.append(1-SSE_3/SST)
  
    predict_BR,test_Bagging = Bagging_reg(standard(X_train),y_train,(X_test-X_train.mean())/(X_train.std()))
    y_predict_BR = pd.concat([y_predict_BR,predict_BR])
    SSE_4 = ((np.array(y_train_1)-np.array(test_Bagging)) ** 2).sum()
    R_BR.append(1-SSE_4/SST)

    predict_GB,test_GB = Gradient_Boosting(standard(X_train),y_train,(X_test-X_train.mean())/(X_train.std()))
    y_predict_GB = pd.concat([y_predict_GB,predict_GB])
    SSE_5 = ((np.array(y_train_1)-np.array(test_GB)) ** 2).sum()
    R_GB.append(1-SSE_5/SST)

    predict_RF,test_RF = RF_reg(standard(X_train),y_train,(X_test-X_train.mean())/(X_train.std()))
    y_predict_RF = pd.concat([y_predict_RF,predict_RF])
    SSE_6 = ((np.array(y_train_1)-np.array(test_RF)) ** 2).sum()
    R_RF.append(1-SSE_6/SST)

    predict_LGBM,test_LGBM = LGBM(standard(X_train),y_train,(X_test-X_train.mean())/(X_train.std()))
    y_predict_LGBM = pd.concat([y_predict_LGBM,predict_LGBM])
    SSE_7 = ((np.array(y_train_1)-np.array(test_LGBM)) ** 2).sum()
    R_LGBM.append(1-SSE_7/SST)

    day = pd.concat([day,test_date])
    y_true = pd.concat([y_true,y_test]) 

# In[54]:


y_true.index = [j for j in range(len(y_true))]

# In[55]:

y_predict_LGBM.index = [j for j in range(len(y_predict_LGBM))]

y_predict_lasso.index = [j for j in range(len(y_predict_lasso))]

y_predict_GB.index = [j for j in range(len(y_predict_GB))]

y_predict_BR.index = [j for j in range(len(y_predict_BR))]

y_predict_lin.index = [j for j in range(len(y_predict_lin))]

y_predict_RF.index = [j for j in range(len(y_predict_RF))]
# In[58]:



# In[ ]:
r2_lin = r2_score(y_true,y_predict_lin)
r_lin = np.array(R_lin).mean()
print 'OLS OOSR2:'
print r2_lin
print 'OLS ISR2'
print r_lin

r2_ridge = r2_score(y_true,y_predict_ridge)
r_ridge = np.array(R_ridge).mean()
print 'ridge OOSr2:'
print r2_ridge
print 'ridge ISR2'
print r_ridge

r2_lasso = r2_score(y_true,y_predict_lasso)
r_lasso = np.array(R_lasso).mean()
print 'lasso OOSr2:'
print r2_lasso
print 'lasso ISr2:'
print r_lasso

r2_BR = r2_score(y_true,y_predict_BR)
r_BR = np.array(R_BR).mean()
print 'BR OOSr2:'
print r2_BR
print 'BR ISr2'
print r_BR

r2_GB = r2_score(y_true,y_predict_GB)
r_GB = np.array(R_GB).mean()
print 'GB OOSr2:'
print r2_GB
print 'GB ISr2'
print r_GB

r2_RF = r2_score(y_true,y_predict_RF)
r_RF = np.array(R_RF).mean()
print 'RF r2:'
print r2_RF
print 'RF ISr2'
print r_RF

r2_LGBM = r2_score(y_true,y_predict_LGBM)
r_LGBM = np.array(R_LGBM).mean()
print 'LGBM OOSr2:'
print r2_LGBM
print 'LGBM ISr2'
print r_LGBM

r_squared= pd.DataFrame([str(r2_lin*100)+'%',str(r2_ridge*100)+'%',
                         str(r2_lasso*100)+'%',str(r2_GB*100)+'%',
                         str(r2_BR*100)+'%',str(r2_RF*100)+'%',str(r2_LGBM*100)+'%'],
                         columns=['OOS R^2'],
                         index=['OLS','Ridge','Lasso','GBR','BR','RF','LGBM'])
r_squared.to_csv('oosr2.csv')

r_squared_in= pd.DataFrame([str(r_lin*100)+'%',str(r_ridge*100)+'%',
                         str(r_lasso*100)+'%',str(r_GB*100)+'%',
                         str(r_BR*100)+'%',str(r_RF*100)+'%',str(r_LGBM*100)+'%'],
                         columns=['IS R^2'],
                         index=['OLS','Ridge','Lasso','GBR','BR','RF','LGBM'])
r_squared_in.to_csv('isr2.csv')

df_lin = pd.DataFrame(columns = ['date','y_predict_ols','y_true'])
df_lin['date'] = day.iloc[:,0]
df_lin['y_predict_ols'] = y_predict_lin.values
df_lin['y_true'] = y_true.values
print 'ols corr:',df_lin.corr()
df_lin.to_csv('y_predict_ols.csv',index=False)

df_ridge = pd.DataFrame(columns = ['date','y_predict_ridge','y_true'])
df_ridge['date'] = day.iloc[:,0]
df_ridge['y_predict_ridge'] = y_predict_ridge.values
df_ridge['y_true'] = y_true.values
print 'ridge corr:',df_ridge.corr()
df_ridge.to_csv('y_predict_ridge.csv',index=False)


df_lasso = pd.DataFrame(columns = ['date','y_predict_lasso','y_true'])
df_lasso['date'] = day.iloc[:,0]
df_lasso['y_predict_lasso'] = y_predict_lasso.values
df_lasso['y_true'] = y_true.values
print 'lasso corr:'
print df_lasso.corr()
df_lasso.to_csv('y_predict_lasso.csv',index=False)

df_BR = pd.DataFrame(columns = ['date','y_predict_BR','y_true'])
df_BR['date'] = day.iloc[:,0]
df_BR['y_predict_BR'] = y_predict_BR.values
df_BR['y_true'] = y_true.values
print 'BR corr:'
print df_BR.corr()
df_BR.to_csv('y_predict_BR.csv',index=False)

df_GB = pd.DataFrame(columns = ['date','y_predict_GB','y_true'])
df_GB['date'] = day.iloc[:,0]
df_GB['y_predict_GB'] = y_predict_GB.values
df_GB['y_true'] = y_true.values
print 'GB corr:'
print df_GB.corr()
df_GB.to_csv('y_predict_GB.csv',index=False)

df_RF = pd.DataFrame(columns = ['date','y_predict_RF','y_true'])
df_RF['date'] = day.iloc[:,0]
df_RF['y_predict_RF'] = y_predict_RF.values
df_RF['y_true'] = y_true.values
print 'RF corr:'
print df_RF.corr()
df_RF.to_csv('y_predict_RF.csv',index=False)

df_LGBM = pd.DataFrame(columns = ['date','y_predict_LGBM','y_true'])
df_LGBM['date'] = day.iloc[:,0]
df_LGBM['y_predict_LGBM'] = y_predict_LGBM.values
df_LGBM['y_true'] = y_true.values
print 'LGBM corr:'
print df_LGBM.corr()
df_LGBM.to_csv('y_predict_LGBM.csv',index=False)
