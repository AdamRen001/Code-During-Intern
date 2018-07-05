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

from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
def LGBMR_CV(X,y,X_test):
    estimator = LGBMRegressor(n_jobs=-1)
    param_grid = {
	          'boosting_type':['gbdt','dart','goss'],
	          'n_estimators':[100,200,300],
	          'learning_rate': [0.05,0.5,1],
                  'reg_alpha': [0.05,0.5,1],
		}
    lgbm_reg = GridSearchCV(estimator,param_grid)
    lgbm_reg.fit(X,y)
    y_lgbm_pred = lgbm_reg.predict(X_test)
    pred_lgbm = pd.DataFrame(y_lgbm_pred)
    print lgbm_reg.best_params_
    return pred_lgbm,lgbm_reg.best_params_
def LGBM_reg(X,y,X_test):
    lgbm_reg = LGBMRegressor(n_jobs=-1,n_estimators=200,
                             boosting_type='dart',learning_rate=0.05,
                             reg_aplha=0.5)
    lgbm_reg.fit(X, y)
    y_lgbm_pred = lgbm_reg.predict(X_test)
    pred_lgbm = pd.DataFrame(y_lgbm_pred)
    return pred_lgbm

y_sign = pd.read_csv('ypredict_lgbm.csv')
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
index = len(d1) + len(d2) + len(d3) + len(d4) + len(d5)+len(d6)
print ('index =',index)
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
pred_lgbr_pos,best_estimator_pos = LGBMR_CV(X1_pos,y1_pos,X_test_pos)
pred_lgbr_neg,best_estimator_neg = LGBMR_CV(X1_neg,y1_neg,X_test_neg)

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
    pred_lgbr_pos = pd.concat([pred_lgbr_pos,LGBM_reg(X_pos,y_pos,X_test_pos)])
    pred_lgbr_neg = pd.concat([pred_lgbr_neg,LGBM_reg(X_neg,y_neg,X_test_neg)])

pred_lgbr = pd.concat([pred_lgbr_pos,pred_lgbr_neg])
from sklearn.metrics import r2_score
r2_lgbr = r2_score(y_test,pred_lgbr)
print ('R^2 lgbr=',r2_lgbr)
method5 = pd.DataFrame({'TradingDay':tradingday.values,
                        'UpdateTime':updatetime.values,
                        'y_predict':pred_lgbr.iloc[:,0]
                       })
method5.sort_values(by=['TradingDay','UpdateTime'],inplace=True)
method5.to_csv('/home/rzg/ypredict_LGBMR_6.csv',index=False)
