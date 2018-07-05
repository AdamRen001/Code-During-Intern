import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


# In[3]:


def daily_backtest(price,predicted,threshold):
    pos=[0]
    account=[0]
    trade=[]
    Num = 0
    for i in range(1,len(price)):
        signal = predicted[i]
        if pos[-1]==1 and signal<0:
            pos.append(0)
            account.append(price[i]-price[i-1])
            trade.append(price[i]-p)
            Num = Num + 1
        elif pos[-1]==0 and signal>=threshold:
            pos.append(1)
            account.append(0)
            p = price[i]
            Num = Num + 1
        elif pos[-1]==0 and signal<=-threshold:
            pos.append(-1)
            account.append(0)
            p = price[i]
            Num = Num + 1
        elif pos[-1]==1 and signal>0:
            pos.append(1)
            account.append(price[i]-price[i-1])
        elif pos[-1]==-1 and signal<0:
            pos.append(-1)
            account.append(price[i-1]-price[i])
        elif pos[-1]==-1 and signal>0:
            pos.append(0)
            account.append(price[i-1]-price[i])
            trade.append(p-price[i])
            Num = Num + 1
        else:
            pos.append(0)
            account.append(0)
    return pd.DataFrame({'pos':pos,'account':account}),pd.DataFrame(trade),Num


alldata = pd.read_csv('/home/rzg/alldata.csv')
D01_close = alldata[['date','XS00000489']]
D01_close.columns = ['date','Close']

ols = pd.read_csv('/home/rzg/D000001/y_predict_ols.csv')
lasso = pd.read_csv('/home/rzg/D000001/y_predict_lasso.csv')
ridge = pd.read_csv('/home/rzg/D000001/y_predict_ridge.csv')
BR = pd.read_csv('/home/rzg/D000001/y_predict_BR.csv')

ols_predict = ols['y_predict_ols']
lasso_predict = lasso['y_predict_lasso']
ridge_predict = ridge['y_predict_ridge'] 
BR_predict = BR['y_predict_BR']

ols_data = pd.merge(ols,D01_close,on=['date'],how='inner')
print ols_data.tail(3)
print "......"
lasso_data = pd.merge(lasso,D01_close,on=['date'],how='inner')
print lasso_data.tail(3)
print "......"
ridge_data = pd.merge(ridge,D01_close,on=['date'],how='inner') 
print ridge_data.tail(3)
print "......"
BR_data = pd.merge(BR,D01_close,on=['date'],how='inner') 
print BR_data.tail(3)
print "......"


#thresholds = [0.0015,0.001,0.0008,0.0006]
#thresholds = [0.005,0.003,0.002,0.0015]
thresholds = [0.001,0.0005,0.0001,0.00005]
sharpe_ols = []
sharpe_ridge = []
sharpe_lasso = []
sharpe_BR = []
TradeNum_ols = []
TradeNum_ridge = []
TradeNum_lasso = []
TradeNum_BR = []
TradeFreq_ols = []
TradeFreq_ridge = []
TradeFreq_lasso = []
TradeFreq_BR = []
for threshold in thresholds:
    pos_account_ols,trade_ols,num_ols = daily_backtest(ols_data['Close'].values,ols_data['y_predict_ols'].values,threshold)
    print ",,,"
    print len(pos_account_ols)
    print pos_account_ols.tail()
    print "*****OLS MODEL***** "
    print "Threshold:",threshold
    print "Number of day:",len(pos_account_ols)
    print "Number of Trade day:", num_ols
    #print "Number of Trade day:",len(trade_ols)
    TradeNum_ols.append(num_ols)
    print "Trade Frequency:",float(num_ols)/float(len(pos_account_ols))
    TradeFreq_ols.append(float(num_ols)/float(len(pos_account_ols)))
    sharpe_ols.append(pos_account_ols['account'].values.mean()/pos_account_ols['account'].values.std())

    pos_account_ridge,trade_ridge,num_ridge = daily_backtest(ridge_data['Close'].values,ridge_data['y_predict_ridge'].values,threshold)
    print ",,,"
    print len(pos_account_ridge)
    print pos_account_ridge.tail()
    print "*****Ridge MODEL***** "
    print "Threshold:",threshold
    print "Number of day:",len(pos_account_ridge)
    print "Number of Trade day:",num_ridge
    TradeNum_ridge.append(num_ridge)
    print "Trade Frequency:",float(num_ridge)/float(len(pos_account_ridge))
    TradeFreq_ridge.append(float(num_ridge)/float(len(pos_account_ridge)))

    sharpe_ridge.append(pos_account_ridge['account'].values.mean()/pos_account_ridge['account'].values.std())

    pos_account_lasso,trade_lasso,num_lasso = daily_backtest(lasso_data['Close'].values,lasso_data['y_predict_lasso'].values,threshold)
    print ",,,"
    print len(pos_account_lasso)
    print pos_account_lasso.tail()
    print "*****Lasso MODEL***** "
    print "Threshold:",threshold
    print "Number of day:",len(pos_account_lasso)
    print "Number of Trade day:",num_lasso
    TradeNum_lasso.append(num_lasso)
    print "Trade Frequency:",float(num_lasso)/float(len(pos_account_lasso))
    TradeFreq_lasso.append(float(num_lasso)/float(len(pos_account_lasso)))

    sharpe_lasso.append(pos_account_lasso['account'].values.mean()/pos_account_lasso['account'].values.std())

    pos_account_BR,trade_BR,num_BR = daily_backtest(BR_data['Close'].values,BR_data['y_predict_BR'].values,threshold)
    print ",,,"
    print len(pos_account_BR)
    print pos_account_BR.tail()
    print "*****Bagging MODEL***** "
    print "Threshold:",threshold
    print "Number of day:",len(pos_account_BR)
    print "Number of Trade day:",num_BR
    TradeNum_BR.append(num_BR)
    print "Trade Frequency:",float(num_BR)/float(len(pos_account_BR))
    TradeFreq_BR.append(float(num_BR)/float(len(pos_account_BR)))

    sharpe_BR.append(pos_account_BR['account'].values.mean()/pos_account_BR['account'].values.std())
print "sharpe for ols:",sharpe_ols
print "sharpe for ridge:",sharpe_ridge
print "sharpe for lasso:",sharpe_lasso
print "sharpe for Bagging:",sharpe_BR

target_data_ols = pd.DataFrame(columns=['Threshold','NumBacktest','NumTrade','TradeFreq','SharpeRatio'])
target_data_ols['Threshold'] = thresholds
target_data_ols['NumBacktest'] = len(ols_data)
target_data_ols['NumTrade'] = TradeNum_ols
target_data_ols['TradeFreq'] = TradeFreq_ols
target_data_ols['SharpeRatio'] = sharpe_ols
target_data_ols.to_csv('OLS_info.csv',index=False)

target_data_ridge = pd.DataFrame(columns=['Threshold','NumBacktest','NumTrade','TradeFreq','SharpeRatio'])
target_data_ridge['Threshold'] = thresholds
target_data_ridge['NumBacktest'] = len(ridge_data)
target_data_ridge['NumTrade'] = TradeNum_ridge
target_data_ridge['TradeFreq'] = TradeFreq_ridge
target_data_ridge['SharpeRatio'] = sharpe_ridge
target_data_ridge.to_csv('Ridge_info.csv',index=False)
target_data_lasso = pd.DataFrame(columns=['Threshold','NumBacktest','NumTrade','TradeFreq','SharpeRatio'])
target_data_lasso['Threshold'] = thresholds
target_data_lasso['NumBacktest'] = len(lasso_data)
target_data_lasso['NumTrade'] = TradeNum_lasso
target_data_lasso['TradeFreq'] = TradeFreq_lasso
target_data_lasso['SharpeRatio'] = sharpe_lasso
target_data_lasso.to_csv('Lasso_info.csv',index=False)

target_data_BR = pd.DataFrame(columns=['Threshold','NumBacktest','NumTrade','TradeFreq','SharpeRatio'])
target_data_BR['Threshold'] = thresholds
target_data_BR['NumBacktest'] = len(BR_data)
target_data_BR['NumTrade'] = TradeNum_BR
target_data_BR['TradeFreq'] = TradeFreq_BR
target_data_BR['SharpeRatio'] = sharpe_BR
target_data_BR.to_csv('Bagging_info.csv',index=False)

