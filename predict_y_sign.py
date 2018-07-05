
# coding: utf-8

# In[1]:

import re
import os
import numpy as np
import pandas as pd
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

# In[10]:


from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
#SVM 
def SGD_SVM_clf(X,y,X_test):  
    X_train = StandardScaler().fit_transform(X)
    y1 = np.array(y)
    y2 = np.where(y1>0,1,-1)
    svm_clf = SGDClassifier(loss='hinge',
                            penalty='elasticnet',
                            alpha=0.05,n_jobs=-1)
    svm_clf.fit(X_train,y2.ravel())
    y_svm_pred = svm_clf.predict(X_test)
    return pd.DataFrame(y_svm_pred)


# In[3]:


#Logistic Regression
def SGD_Logistic_clf(X,y,X_test):  
    X_train = StandardScaler().fit_transform(X)
    y1 = np.array(y)
    y2 = np.where(y1>0,1,-1)
    logistic_clf = SGDClassifier(loss='log',
                                 penalty='elasticnet',
                                 alpha=0.01,n_jobs=-1)
    logistic_clf.fit(X_train,y2.ravel())
    y_logistic_pred = logistic_clf.predict(X_test)
    return pd.DataFrame(y_logistic_pred)


# In[4]:


#GradientBoosting
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
def GBCV(X,y,x_test):
    estimator = GradientBoostingClassifier()
    y1 = np.array(y)
    y2 = np.where(y1>0,1,-1)
    param_grid = {
	           'loss':['deviance','exponential'],
		   'n_estimators':[100,200],
	           'learning_rate': [0.1,0.3,1],
                   'max_depth': [3, 5,10]
	          }
    gb_clf = GridSearchCV(estimator,param_grid)
    gb_clf.fit(X,y2.ravel())
    y_gb_pred = gb_clf.predict(X_test)
    print pd.DataFrame(y_gb_pred).head(1)
    return pd.DataFrame(y_gb_pred),gb_clf.best_estimator_ 

def GB_clf(X,y,X_test,params):
    y1 = np.array(y)
    y2 = np.where(y1>0,1,-1)
    gb_clf = GradientBoostingClassifier(params)
    gb_clf.fit(X,y2.ravel())
    y_gb_pred = gb_clf.predict(X_test)
    return pd.DataFrame(y_gb_pred)

#Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
def AdaBoost_clf(X,y,X_test):
    X_train = StandardScaler().fit_transform(X)
    y1 = np.array(y)
    y2 = np.where(y1>0,1,-1)
    adaboost_clf = AdaBoostClassifier(LinearSVC(),algorithm='SAMME')
    adaboost_clf.fit(X_train,y2.ravel())
    y_adaboost_pred = adaboost_clf.predict(X_test)
    return pd.DataFrame(y_adaboost_pred)

#BaggingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC
def Bagging_clf(X,y,X_test):
    X_train = StandardScaler().fit_transform(X)
    y1 = np.array(y)
    y2 = np.where(y1>0,1,-1)
    bagging_clf = BaggingClassifier(LinearSVC(),n_estimators=10,n_jobs=-1)
    bagging_clf.fit(X_train,y2.ravel())
    y_bagging_pred = bagging_clf.predict(X_test)
    return pd.DataFrame(y_bagging_pred)

# In[5]:


startdate = '20180104'
enddate = '20180201'
file_paths = get_date_map('/home/tysongroup/lowdim/',startdate,enddate)
file_paths = sorted(file_paths.values())
da = pd.read_csv(file_paths[3],header=None)
y_test = da.iloc[:,len(da.columns)-1]
tradingday = da.iloc[:,0]
updatetime = da.iloc[:,1]
for f in file_paths[4:len(file_paths)]:
    df = pd.read_csv(f,header=None)
    y_test = pd.concat([y_test,df.iloc[:,len(da.columns)-1]])
    tradingday = pd.concat([tradingday,df.iloc[:,0]])
    updatetime = pd.concat([updatetime,df.iloc[:,1]])


# In[6]:


y1 = np.array(y_test)
clf_true = np.where(y1>0,1,-1)
clf_true = pd.DataFrame(clf_true)


# In[ ]:


data1 = pd.read_csv(file_paths[0],header=None)
for f in file_paths[1:3]:
    df1 = pd.read_csv(f,header=None)
    data1 = pd.concat([data1,df1])
y1 = data1.iloc[:,len(data1.columns)-1]
X1 = data1.iloc[:,2:len(data1.columns)-1]
df1 = pd.read_csv(file_paths[3],header=None)
X_test = df1.iloc[:,2:len(data1.columns)-1]
pred_logistic = SGD_Logistic_clf(X1,y1,X_test)
pred_svm = SGD_SVM_clf(X1,y1,X_test)
pred_gb,params = GBCV(X1,y1,X_test)
pred_ada = AdaBoost_clf(X1,y1,X_test)
pred_bagging = Bagging_clf(X1,y1,X_test)

# In[ ]:


for i in range(1,len(file_paths)-3):
    data = pd.read_csv(file_paths[i],header=None)
    for f in file_paths[i+1:i+3]:
        df = pd.read_csv(f,header=None)
        data = pd.concat([data,df])
    y = data.iloc[:,len(data.columns)-1]
    X = (data.iloc[:,2:len(data.columns)-1])
    df = pd.read_csv(file_paths[i+3],header=None)
    X_test = df.iloc[:,2:len(df.columns)-1]
    pred_logistic = pd.concat([pred_logistic,SGD_Logistic_clf(X,y,X_test)])
    pred_svm = pd.concat([pred_svm,SGD_SVM_clf(X,y,X_test)])
    pred_gb = pd.concat([pred_gb,GB_clf(X,y,X_test,params)])
    pred_ada = pd.concat([pred_ada,AdaBoost_clf(X,y,X_test)])
    pred_bagging = pd.concat([pred_bagging,Bagging_clf(X,y,X_test)])    
# In[ ]:


from sklearn.metrics import accuracy_score
ac0 = accuracy_score(clf_true,pred_logistic)
ac1 = accuracy_score(clf_true,pred_svm)
ac2 = accuracy_score(clf_true,pred_gb)
ac3 = accuracy_score(clf_true,pred_ada)
ac4 = accuracy_score(clf_true,pred_bagging)
print('Logistic Accuracy=',ac0)
print('SVM Accuracy=',ac1)
print('GradientBoosting Accuracy=',ac2)
print('AdaBoost Accuracy=',ac3)
print('Bagging Accuracy=',ac4)
AC = pd.DataFrame([str(ac0*100)+'%',str(ac1*100)+'%',
                   str(ac2*100)+'%',str(ac3*100)+'%',
                   str(ac4*100)],
                   columns=['Accuracy'],
                   index=['Logistic','SVM','GradientBoosting','AdaBoost',
                          'Bagging'])

AC.to_csv('/home/rzg/project2/accuracy.csv')
AC = pd.DataFrame([str(ac0*100)+'%',str(ac1*100)+'%',
                   str(ac2*100)+'%'],
                   columns=['Accuracy'],
                   index=['Logistic','SVM','GradientBoosting'])
AC.to_csv('accuracy2.csv')
method1 = pd.DataFrame({'TradingDay':tradingday,
                        'UpdateTime':updatetime,
                        'y_predict':pred_logistic.iloc[:,0]
                       })
method1.to_csv('/home/rzg/project2/ypredict_logistic.csv',index=False)

method2 = pd.DataFrame({'TradingDay':tradingday,
                        'UpdateTime':updatetime,
                        'y_predict':pred_svm.iloc[:,0]
                       })
method2.to_csv('/home/rzg/project2/ypredict_svm.csv',index=False)

method3 = pd.DataFrame({'TradingDay':tradingday,
                        'UpdateTime':updatetime,
                        'y_predict':pred_gb.iloc[:,0]
                       })
method3.to_csv('/home/rzg/project2/ypredict_gb.csv',index=False)
method4 = pd.DataFrame({'TradingDay':tradingday,
                        'UpdateTime':updatetime,
                        'y_predict':pred_ada.iloc[:,0]
                       })
method4.to_csv('/home/rzg/project2/ypredict_adaboost.csv',index=False)

method5 = pd.DataFrame({'TradingDay':tradingday,
                        'UpdateTime':updatetime,
                        'y_predict':pred_bagging.iloc[:,0]
                       })
method5.to_csv('/home/rzg/project2/ypredict_Bagging.csv',index=False)
