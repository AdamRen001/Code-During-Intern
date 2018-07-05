
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import re
import os 
from sklearn import linear_model

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

#n[ ]:

#ols regression
from sklearn.linear_model import LinearRegression
def ols_regression(X,y,X_test):
    linreg = LinearRegression(fit_intercept=False,n_jobs=-1)
    linreg.fit(X,y)
    y_pred = linreg.predict(X_test)
    pred_ols = pd.DataFrame(y_pred)
    print linreg.coef_
    return pred_ols


# In[ ]:
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

# In[ ]:
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

# In[ ]:
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

# In[ ]:
startdate = '20180320'
enddate = '20180413'
file_paths = get_date_map('/data/public/cleandata/featurematrix_obi/',startdate,enddate)
file_paths = sorted(file_paths.values())
data0 = pd.read_csv(file_paths[0])
featurematrix = data0.iloc[:,3:len(data0.columns)-3]
origColumns = featurematrix.columns
weights1 = pd.read_csv('/home/intern/weights/pc1_weights.csv')
cols = weights1.columns
code = []
#select columns
for col in cols:
    match = re.search('(\d{6})', col)
    if match is not None:
        code.append(match.group(1))
selectColumns = ['TradingDay','UpdateTime']
for col in origColumns:
    for name in code:
        if col.find(name)<0:
            continue
        else:
            selectColumns.append(col)


# In[ ]:

#read y_true&date&time
d1 = pd.read_csv(file_paths[3])
tradingday = d1.iloc[:,0]
updatetime = d1.iloc[:,1]
y1_true = d1.iloc[:,len(d1.columns)-3]
y2_true = d1.iloc[:,len(d1.columns)-2]
y3_true = d1.iloc[:,len(d1.columns)-1]
print len(y1_true)
for f in file_paths[4:]:    
    d = pd.read_csv(f)
    print len(d)
    tradingday = pd.concat([tradingday,d.iloc[:,0]])
    updatetime = pd.concat([updatetime,d.iloc[:,1]])
    y1_true = pd.concat([y1_true,d.iloc[:,len(d.columns)-3]])
    y2_true = pd.concat([y2_true,d.iloc[:,len(d.columns)-2]])
    y3_true = pd.concat([y3_true,d.iloc[:,len(d.columns)-1]])


# In[ ]:

#1st 3 days predict, 4th day
data1 = pd.read_csv(file_paths[0])
for f in file_paths[1:3]:
    df = pd.read_csv(f)
    data1 = pd.concat([data1,df])
y1_1 = data1.iloc[:,len(data1.columns)-3]
y2_1 = data1.iloc[:,len(data1.columns)-2]
y3_1 = data1.iloc[:,len(data1.columns)-1]
featurematrix = data1.iloc[:,3:len(data1.columns)-3]
featurematrix = featurematrix[selectColumns]
matrix1 = featurematrix.iloc[:,0]
for i in range(5,len(featurematrix.columns),5):
    matrix1 = pd.concat([matrix1,featurematrix.iloc[:,i]],axis=1)
matrix2 = featurematrix.iloc[:,1]
for i in range(6,len(featurematrix.columns),5):
    matrix2 = pd.concat([matrix2,featurematrix.iloc[:,i]],axis=1)
matrix3 = featurematrix.iloc[:,2]
for i in range(7,len(featurematrix.columns),5):
    matrix3 = pd.concat([matrix3,featurematrix.iloc[:,i]],axis=1)
matrix4 = featurematrix.iloc[:,3]
for i in range(8,len(featurematrix.columns),5):
    matrix4 = pd.concat([matrix4,featurematrix.iloc[:,i]],axis=1)
matrix5 = featurematrix.iloc[:,4]
for i in range(9,len(featurematrix.columns),5):
    matrix5 = pd.concat([matrix5,featurematrix.iloc[:,i]],axis=1)
files2 = sorted(os.listdir('/home/intern/weights/'))
file_paths2 =['/home/intern/weights/'+f for f in files2]
#find features X
X = data1.iloc[:,0:2]
for f in file_paths2:
    weights = pd.read_csv(f)
    x1 = np.dot(matrix1,weights.T); x1 = x1.ravel() 
    x2 = np.dot(matrix2,weights.T); x2 = x2.ravel()
    x3 = np.dot(matrix3,weights.T); x3 = x3.ravel()
    x4 = np.dot(matrix4,weights.T); x4 = x4.ravel()
    x5 = np.dot(matrix5,weights.T); x5 = x5.ravel()
    print x1
    print x1.shape
    x1 = pd.Series(x1)
    x2 = pd.Series(x2)
    x3 = pd.Series(x3)
    x4 = pd.Series(x4)
    x5 = pd.Series(x5)
    X = pd.concat([X,x1,x2,x3,x4,x5],axis=1)
from IFStock import ifstockupdate
startdate = '20180320'
enddate = '20180413'
file_paths3 = get_date_map('/data/public/cleandata/StockIXcsv/',startdate,enddate)
file_paths3 = sorted(file_paths.values())

df1 = pd.read_csv(file_paths[3])
featurematrix = df1.iloc[:,3:len(df1.columns)-3]
featurematrix = featurematrix[selectColumns]
matrix1 = featurematrix.iloc[:,0]
for i in range(5,len(featurematrix.columns),5):
    matrix1 = pd.concat([matrix1,featurematrix.iloc[:,i]],axis=1)
matrix2 = featurematrix.iloc[:,1]
for i in range(6,len(featurematrix.columns),5):
    matrix2 = pd.concat([matrix2,featurematrix.iloc[:,i]],axis=1)
matrix3 = featurematrix.iloc[:,2]
for i in range(7,len(featurematrix.columns),5):
    matrix3 = pd.concat([matrix3,featurematrix.iloc[:,i]],axis=1)
matrix4 = featurematrix.iloc[:,3]
for i in range(8,len(featurematrix.columns),5):
    matrix4 = pd.concat([matrix4,featurematrix.iloc[:,i]],axis=1)
matrix5 = featurematrix.iloc[:,4]
for i in range(9,len(featurematrix.columns),5):
    matrix5 = pd.concat([matrix5,featurematrix.iloc[:,i]],axis=1)
X_test = pd.DataFrame()
for f in file_paths2:
    weights = pd.read_csv(f)
    x1 = np.dot(matrix1,weights.T); x1 = x1.ravel() 
    x2 = np.dot(matrix2,weights.T); x2 = x2.ravel() 
    x3 = np.dot(matrix3,weights.T); x3 = x3.ravel() 
    x4 = np.dot(matrix4,weights.T); x4 = x4.ravel() 
    x5 = np.dot(matrix5,weights.T); x5 = x5.ravel() 
    x1 = pd.Series(x1)
    x2 = pd.Series(x2)
    x3 = pd.Series(x3)
    x4 = pd.Series(x4)
    x5 = pd.Series(x5)
    X_test = pd.concat([X_test,x1,x2,x3,x4,x5],axis=1)


# In[ ]:

pred_ols_1 = ols_regression(X,y1_1,X_test)
pred_ols_2 = ols_regression(X,y2_1,X_test)
pred_ols_3 = ols_regression(X,y3_1,X_test)
print "len X_test=", len(X_test)
print "len ols_1=",len(pred_ols_1)
"""
# In[ ]:

pred_ridge_1,best_alpha1_1 = ridge_regression(X,y1_1,X_test)
pred_ridge_2,best_alpha1_2 = ridge_regression(X,y2_1,X_test)
pred_ridge_3,best_alpha1_3 = ridge_regression(X,y3_1,X_test)


# In[ ]:

pred_lasso_1,best_alpha2_1 = lasso_regression(X,y1_1,X_test)
pred_lasso_2,best_alpha2_2 = lasso_regression(X,y2_1,X_test)
pred_lasso_3,best_alpha2_3 = lasso_regression(X,y3_1,X_test)


# In[ ]:

pred_elastic_1,best_alpha3_1,best_l1_ratio_1 = elastic_regression(X,y1_1,X_test)
pred_elastic_2,best_alpha3_2,best_l1_ratio_2 = elastic_regression(X,y2_1,X_test)
pred_elastic_3,best_alpha3_3,best_l1_ratio_3 = elastic_regression(X,y3_1,X_test)


# In[ ]:

pred_GB_1,best_estimator_1 = GBRCV(X,y1_1,X_test)
pred_GB_2,best_estimator_2 = GBRCV(X,y2_1,X_test)
pred_GB_3,best_estimator_3 = GBRCV(X,y3_1,X_test)

"""
# In[ ]:

for i in range(1,len(file_paths)-3):
    data = pd.read_csv(file_paths[i])
    for f in file_paths[i+1:i+3]:
        df = pd.read_csv(f)
        data = pd.concat([data,df])
    featurematrix = data.iloc[:,3:len(data.columns)-3]
    featurematrix = featurematrix[selectColumns] 
    y1 = data.iloc[:,len(data.columns)-3]
    y2 = data.iloc[:,len(data.columns)-2]
    y3 = data.iloc[:,len(data.columns)-1]
    
    matrix1 = featurematrix.iloc[:,0]
    for j in range(5,len(featurematrix.columns),5):
        matrix1 = pd.concat([matrix1,featurematrix.iloc[:,j]],axis=1)
    matrix2 = featurematrix.iloc[:,1]
    for j in range(6,len(featurematrix.columns),5):
        matrix2 = pd.concat([matrix2,featurematrix.iloc[:,j]],axis=1)
    matrix3 = featurematrix.iloc[:,2]
    for j in range(7,len(featurematrix.columns),5):
        matrix3 = pd.concat([matrix3,featurematrix.iloc[:,j]],axis=1)
    matrix4 = featurematrix.iloc[:,3]
    for j in range(8,len(featurematrix.columns),5):
        matrix4 = pd.concat([matrix4,featurematrix.iloc[:,j]],axis=1)
    matrix5 = featurematrix.iloc[:,4]
    for j in range(9,len(featurematrix.columns),5):
        matrix5 = pd.concat([matrix5,featurematrix.iloc[:,j]],axis=1)
    files2 = sorted(os.listdir('/home/intern/weights/'))
    file_paths2 =['/home/intern/weights/'+f for f in files2]
    #find features X
    X = pd.DataFrame()
    for f in file_paths2:
        weights = pd.read_csv(f)
        x1 = np.dot(matrix1,weights.T); x1 = x1.ravel() 
        x2 = np.dot(matrix2,weights.T); x2 = x2.ravel() 
        x3 = np.dot(matrix3,weights.T); x3 = x3.ravel() 
        x4 = np.dot(matrix4,weights.T); x4 = x4.ravel() 
        x5 = np.dot(matrix5,weights.T); x5 = x5.ravel() 
        print (x1.shape) 
        x1 = pd.Series(x1)
        x2 = pd.Series(x2)
        x3 = pd.Series(x3)
        x4 = pd.Series(x4)
        x5 = pd.Series(x5)
        X = pd.concat([X,x1,x2,x3,x4,x5],axis=1)
    #gain X_test
    df1 = pd.read_csv(file_paths[i+3])
    featurematrix1 = df1.iloc[:,3:len(df1.columns)-3]
    featurematrix1 = featurematrix1[selectColumns]
    matrix1 = featurematrix1.iloc[:,0]
    for j in range(5,len(featurematrix1.columns),5):
        matrix1 = pd.concat([matrix1,featurematrix1.iloc[:,j]],axis=1)
    matrix2 = featurematrix1.iloc[:,1]
    for j in range(6,len(featurematrix1.columns),5):
        matrix2 = pd.concat([matrix2,featurematrix1.iloc[:,j]],axis=1)
    matrix3 = featurematrix1.iloc[:,2]
    for j in range(7,len(featurematrix1.columns),5):
        matrix3 = pd.concat([matrix3,featurematrix1.iloc[:,j]],axis=1)
    matrix4 = featurematrix1.iloc[:,3]
    for j in range(8,len(featurematrix1.columns),5):
        matrix4 = pd.concat([matrix4,featurematrix1.iloc[:,j]],axis=1)
    matrix5 = featurematrix1.iloc[:,4]
    for j in range(9,len(featurematrix1.columns),5):
        matrix5 = pd.concat([matrix5,featurematrix1.iloc[:,j]],axis=1)
    X_test = pd.DataFrame()
    for f in file_paths2:
        weights = pd.read_csv(f)
        x1 = np.dot(matrix1,weights.T); x1 = x1.ravel() 
        x2 = np.dot(matrix2,weights.T); x2 = x2.ravel() 
        x3 = np.dot(matrix3,weights.T); x3 = x3.ravel() 
        x4 = np.dot(matrix4,weights.T); x4 = x4.ravel() 
        x5 = np.dot(matrix5,weights.T); x5 = x5.ravel() 
        x1 = pd.Series(x1)
        x2 = pd.Series(x2)
        x3 = pd.Series(x3)
        x4 = pd.Series(x4)
        x5 = pd.Series(x5)
        X_test = pd.concat([X_test,x1,x2,x3,x4,x5],axis=1)
    pred_ols_1 = pd.concat([pred_ols_1,
                            ols_regression(X,y1,X_test)])
    pred_ols_2 = pd.concat([pred_ols_2,
                            ols_regression(X,y2,X_test)])
    pred_ols_3 = pd.concat([pred_ols_3,
                            ols_regression(X,y3,X_test)])
  
    """
    pred_ridge_1 = pd.concat([pred_ridge_1,
                              ridge(X,y1,X_test,best_alpha1_1)])
    pred_ridge_2 = pd.concat([pred_ridge_2,
                              ridge(X,y2,X_test,best_alpha1_2)])
    pred_ridge_3 = pd.concat([pred_ridge_3,
                              ridge(X,y3,X_test,best_alpha1_3)])
        
    pred_lasso_1 = pd.concat([pred_lasso_1,
                              lasso(X,y1,X_test,best_alpha2_1)])
    pred_lasso_2 = pd.concat([pred_lasso_2,
                              lasso(X,y2,X_test,best_alpha2_2)])
    pred_lasso_3 = pd.concat([pred_lasso_3,
                              lasso(X,y3,X_test,best_alpha2_3)])
        
    pred_elastic_1 = pd.concat([pred_elastic_1,
                                elastic(X,y1,X_test,best_alpha3_1,best_l1_ratio_1)])
    pred_elastic_2 = pd.concat([pred_elastic_2,
                                elastic(X,y2,X_test,best_alpha3_2,best_l1_ratio_2)])
    pred_elastic_3 = pd.concat([pred_elastic_3,
                                elastic(X,y3,X_test,best_alpha3_3,best_l1_ratio_3)])
        
    pred_GB_1 = pd.concat([pred_GB_1,Gradient_Boosting(X,y1,X_test,best_estimator_1 )])
    pred_GB_2 = pd.concat([pred_GB_2,Gradient_Boosting(X,y2,X_test,best_estimator_2 )])
    pred_GB_3 = pd.concat([pred_GB_3,Gradient_Boosting(X,y3,X_test,best_estimator_3 )])
        
    """

# In[ ]:

from sklearn.metrics import r2_score
print len(pred_ols_1)
r2_ols_1 = r2_score(y1_true,pred_ols_1)
print r2_ols_1
"""
r2_ridge_1 = r2_score(y1_true,pred_ridge_1)
r2_lasso_1 = r2_score(y1_true,pred_lasso_1)
r2_elastic_1 = r2_score(y1_true,pred_elastic_1)
r2_GBR_1 = r2_score(y1_true,pred_GB_1)
r_squared_1 = pd.DataFrame([str(r2_ols_1*100)+'%',str(r2_ridge_1*100)+'%',
                            str(r2_lasso_1*100)+'%',str(r2_elastic_1*100)+'%',
                            str(r2_GBR_1*100)+'%'],
                            columns=['Y1 R^2'],
                            index=['OLS','Ridge','Lasso','ElasticNet','GBR'])
"""
r2_ols_2 = r2_score(y2_true,pred_ols_2)
print r2_ols_2
"""
r2_ridge_2 = r2_score(y2_true,pred_ridge_2)
r2_lasso_2 = r2_score(y2_true,pred_lasso_2)
r2_elastic_2 = r2_score(y2_true,pred_elastic_2)
r2_GBR_2 = r2_score(y2_true,pred_GB_2)
r_squared_2 = pd.DataFrame([str(r2_ols_2*100)+'%',str(r2_ridge_2*100)+'%',
                            str(r2_lasso_2*100)+'%',str(r2_elastic_2*100)+'%',
                            str(r2_GBR_2*100)+'%'],
                            columns=['Y2 R^2'],
                            index=['OLS','Ridge','Lasso','ElasticNet','GBR'])
"""
r2_ols_3 = r2_score(y3_true,pred_ols_3)
print r2_ols_3
"""
r2_ridge_3 = r2_score(y3_true,pred_ridge_3)
r2_lasso_3 = r2_score(y3_true,pred_lasso_3)
r2_elastic_3 = r2_score(y3_true,pred_elastic_3)
r2_GBR_3 = r2_score(y3_true,pred_GB_3)
r_squared_3 = pd.DataFrame([str(r2_ols_3*100)+'%',str(r2_ridge_3*100)+'%',
                            str(r2_lasso_3*100)+'%',str(r2_elastic_3*100)+'%',
                            str(r2_GBR_3*100)+'%'],
                            columns=['Y3 R^2'],
                            index=['OLS','Ridge','Lasso','ElasticNet','GBR'])


# In[ ]:

r_squared_1.to_csv('y1_R2.csv')
r_squared_2.to_csv('y2_R2.csv')
r_squared_3.to_csv('y3_R2.csv')


# In[ ]:

y1_predict = pd.DataFrame({'TradingDay':tradingday,
                           'UpdateTime':updatetime,
                           'y1_ols_pred':pred_ols_1.iloc[:,0],
                           'y1_ridge_pred':pred_ridge_1.iloc[:,0],
                           'y1_lasso_pred':pred_lasso_1.iloc[:,0],
                           'y1_elastic_pred':pred_elastic_1.iloc[:,0],
                           'y1_GBR_pred':pred_GB_1.iloc[:,0]
                          })
y1_predict.to_csv('y1_predict.csv',index=False)


# In[ ]:

y2_predict = pd.DataFrame({'TradingDay':tradingday,
                           'UpdateTime':updatetime,
                           'y2_ols_pred':pred_ols_2.iloc[:,0],
                           'y2_ridge_pred':pred_ridge_2.iloc[:,0],
                           'y2_lasso_pred':pred_lasso_2.iloc[:,0],
                           'y2_elastic_pred':pred_elastic_2.iloc[:,0],
                           'y2_GBR_pred':pred_GB_2.iloc[:,0]
                          })
y2_predict.to_csv('y2_predict.csv',index=False)


# In[ ]:

y3_predict = pd.DataFrame({'TradingDay':tradingday,
                           'UpdateTime':updatetime,
                           'y3_ols_pred':pred_ols_3.iloc[:,0],
                           'y3_ridge_pred':pred_ridge_3.iloc[:,0],
                           'y3_lasso_pred':pred_lasso_3.iloc[:,0],
                           'y3_elastic_pred':pred_elastic_3.iloc[:,0],
                           'y3_GBR_pred':pred_GB_3.iloc[:,0]
                          })
y3_predict.to_csv('y3_predict.csv',index=False)
"""
r_squared = pd.DataFrame([str(r2_ols_1*100)+'%',
                          str(r2_ols_2*100)+'%',
                          str(r2_ols_3*100)+'%'],
                          columns=['OLS R^2'],
                          index=['Y1','Y2','Y3'])
r_squared.to_csv('r_squared.csv')
y_predict = pd.DataFrame({'TradingDay':tradingday,
                          'UpdateTime':updatetime,
                          'y1_ols_pred':pred_ols_1.iloc[:,0],
                          'y2_ols_pred':pred_ols_2.iloc[:,0],
                          'y3_ols_pred':pred_ols_3.iloc[:,0]
                          })
y_predict.to_csv('y_predict.csv',index=False)

