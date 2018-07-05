key = True
import re
import sys
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor

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

def score(y1,y2):
    return 1-(((y1-y2)**2).sum())/((y2-y2.mean())**2).sum()

def standard(x):
    return (x-x.mean())/x.std()

def Lasso_Reg(X,y,alpha):
    lasso_reg = Lasso(fit_intercept=True,alpha=alpha).fit(X,y)
    return lasso_reg
dfs = list()
select = []
#read config.txt, set parameters
configFileName = 'config.txt'
configFile = open(configFileName, 'r')
configLines = configFile.readlines()
configFile.close()
#csvPath
csvPath = configLines[0].strip()
#startdate
startdate = configLines[1].strip()
#enddate
enddate = configLines[2].strip()
#alphas
alphas = [float(x) for x in configLines[3].strip().split(',')] 
#whether output colnames
TF = bool(configLines[4].strip())
#y_predict output path
y_outputPath = configLines[5].strip() 
#OOR2 output path
R2_outputPath = configLines[6].strip()

file_paths = get_date_map(csvPath,startdate,enddate)
file_paths = sorted(file_paths.values())
R2 = pd.DataFrame()
for alpha in alphas:
    flag = 1
    for i in range(len(file_paths)-3):
        feature = pd.read_csv(file_paths[i])
        names = feature.columns
        feature.columns = [q for q in range(len(feature.columns))]
        for j in range(i+1, i+3):
            feature_temp = pd.read_csv(file_paths[j])
            feature_temp.columns = [q for q in range(len(feature_temp.columns))]
            feature = feature.append(feature_temp)
        forecast = pd.read_csv(file_paths[i+3])
        forecast.columns = [q for q in range(len(forecast.columns))]
        X = feature[range(2,len(feature.columns)-1)]
        y = feature[len(feature.columns)-1]
        lasso_reg = Lasso_Reg(standard(X),y,alpha)
        #select.append([sum(lasso_003.coef_!=0), sum(lasso_004.coef_!=0), sum(lasso_006.coef_!=0),sum(lasso_008.coef_!=0)])
        selected = np.array(np.where(lasso_reg.coef_!=0), dtype = 'int')+2
        selectcol = names[selected]
        if TF:
            xselect = selectcol.tolist()
            ff = open('Lasso_xselectTh%e_%s.txt'%(alpha,file_paths[i+3][-12:-4]),"w")
	    for z in range(len(xselect)):
                ff.write('\n'.join(xselect[z]))
	    ff.close()
        Y_forecast_true = forecast[len(forecast.columns)-1]
        X_forecast = forecast[range(2, len(forecast.columns)-1)]
        Y_predict_lasso = lasso_reg.predict((X_forecast-X.mean())/X.std())
        df = pd.DataFrame(columns = ['TradingDay','UpdateTime','Y_predict_lasso_0'+str(alpha)[2:]], index = forecast.index)
        df['Y_predict_lasso_0'+str(alpha)[2:]] = Y_predict_lasso
        df['TradingDay'] = forecast[0]
        df['UpdateTime'] = forecast[1]
        dfs.append(df) 
	
        if key:
            df_predict = pd.DataFrame(columns = ['TradingDay','UpdateTime','Y_predict_lasso_0'+str(alpha)[2:]], index = forecast.index)
            df_predict['Y_predict_lasso_0'+str(alpha)[2:]] = Y_predict_lasso
            df_predict['TradingDay'] = forecast[0]
            df_predict['UpdateTime'] = forecast[1]
        if flag:
            df_predict.to_csv(y_outputPath[:-4]+'_%e'%alpha+'.csv',index = False)
        else:
            temp_predict = pd.read_csv(y_outputPath[:-4]+'_%e'%alpha+'.csv')
            df_predict = temp_predict.append(df_predict)
            df_predict.to_csv(y_outputPath[:-4]+'_%e'%alpha+'.csv',index = False)
        flag = 0
    y_predict = pd.concat(dfs, ignore_index = True)
    R = pd.DataFrame(index = ['OOSR2'], columns = ['Lasso_0'+str(alpha)[2:]])
    R.loc['OOSR2']['Lasso_0'+str(alpha)[2:]] = score(Y_predict_lasso, Y_forecast_true)
    R2 = pd.concat([R2,R],axis=1)
R2.to_csv(R2_outputPath)
