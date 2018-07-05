
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[96]:


def ask_detect(orderbook,askPriceTrade,bidPriceTrade,threshold,ticks):
    suspect=[[0,0,0,0]]
    temp=0
    for i in range(1,len(orderbook)):
        if suspect[-1][-1]==1:
            temp=1
        for j in range(0,5):
            if orderbook.iloc[i,4*j+3]==orderbook.iloc[i-1,3] and orderbook.iloc[i,4*j+4]-orderbook.iloc[i-1,4]>=threshold            and suspect[-1][-1]==0:
                suspect.append([orderbook.iloc[i,1],orderbook.iloc[i-1,3],orderbook.iloc[i,4*j+4],i,1])
                 
            if suspect[-1][-1]==1 and suspect[-1][1]==orderbook.iloc[i,4*j+3]:
                temp=0
                if suspect[-1][-3]-orderbook.iloc[i,4*j+4]>max(askPriceTrade[i],threshold):
                    suspect[-1][-1]=i-1
                    suspect[-1].append(askPriceTrade[i])
                    suspect[-1].append(sum(bidPriceTrade[suspect[-1][3]:i-1]))
                    suspect[-1].append(0)
                elif i-suspect[-1][-2]>ticks:
                    suspect.pop()
                else:
                    suspect[-1][-3]=orderbook.iloc[i,4*j+4]
        if suspect[-1][-1]==1 and temp==1 and suspect[-1][-3]>max(askPriceTrade[i-1],threshold):
            suspect[-1][-1]=i-1
            suspect[-1].append(askPriceTrade[i-1])
	    suspect[-1].append(sum(bidPriceTrade[suspect[-1][3]:i-1]))
            suspect[-1].append(0)
            
    suspect.remove([0,0,0,0])
    suspect=pd.DataFrame(suspect)
    suspect.columns=['time','price','orderVolume','start','end','tradeVolume','xiaoDan','sig']
    
    return suspect


# In[89]:


def bid_detect(orderbook,bidPriceTrade,askPriceTrade,threshold,ticks):
    suspect=[[0,0,0,0]]
    temp=0
    for i in range(1,len(orderbook)):#
        if suspect[-1][-1]==1:
            temp=1
        for j in range(0,5):
            if orderbook.iloc[i,4*j+5]==orderbook.iloc[i-1,5] and orderbook.iloc[i,4*j+6]-orderbook.iloc[i-1,6]>=threshold            and suspect[-1][-1]==0:
                suspect.append([orderbook.iloc[i,1],orderbook.iloc[i-1,5],orderbook.iloc[i,4*j+6],i,1])
                 
            if suspect[-1][-1]==1 and suspect[-1][1]==orderbook.iloc[i,4*j+5]:
                temp=0
                if suspect[-1][-3]-orderbook.iloc[i,4*j+6]>max(bidPriceTrade[i],threshold):
                    #i
                    suspect[-1][-1]=i-1
                    suspect[-1].append(bidPriceTrade[i])
                    suspect[-1].append(sum(askPriceTrade[suspect[-1][3]:i]))
                    suspect[-1].append(0)
                
                elif i-suspect[-1][-2]>ticks:
                    suspect.pop()
                else:
                    suspect[-1][-3]=orderbook.iloc[i,4*j+6]
        if suspect[-1][-1]==1 and temp==1:
            if suspect[-1][-3]>max(threshold,bidPriceTrade[i-1]):
                suspect[-1][-1]=i-1
                suspect[-1].append(bidPriceTrade[i-1])
		suspect[-1].append(sum(askPriceTrade[suspect[-1][3]:i-1]))
                suspect[-1].append(0)
            else:
                suspect.pop()
    
    suspect.remove([0,0,0,0])
    suspect=pd.DataFrame(suspect)
    suspect.columns=['time','price','orderVolume','start','end','tradeVolume','xiaoDan','sig']
    
    return suspect


# In[82]:


def ask_detect_R(orderbook,askPriceTrade,threshold1,threshold2,ticks):
    suspect=[]
    ask_dict=[{}]
    for j in range(0,5):
            ask_dict[-1][orderbook.iloc[0,4*j+3]]=orderbook.iloc[0,4*j+4]
    for i in range(0,len(orderbook)-1):
        ask_dict.append({})
        for j in range(0,5):
            ask_dict[-1][orderbook.iloc[i+1,4*j+3]]=orderbook.iloc[i+1,4*j+4]
        if orderbook.iloc[i,3] in ask_dict[-1]:
            if orderbook.iloc[i,4]-ask_dict[-1][orderbook.iloc[i,3]]>max(threshold1,askPriceTrade[i]):
                diff1=orderbook.iloc[i,4]-ask_dict[-1][orderbook.iloc[i,3]]
                t=[1,i]
                s=0
                for k in range(i,i-ticks,-1):
                    try:
                        diff2=ask_dict[k][orderbook.iloc[i,3]]-ask_dict[k-1][orderbook.iloc[i,3]]+askPriceTrade[i-1]
                    except:
                        break
                    if diff2>threshold2:
                        s=s+diff2
                        t.append(k)
                        #k-(k-1)
                    if s>=diff1:
                        t[0]=0
                        break
                '''if t[0]==0:
                    suspect.append(t)'''
                suspect.append(t)
        else:
            if orderbook.iloc[i,4]>max(threshold1,askPriceTrade[i]):
                diff1=orderbook.iloc[i,4]-askPriceTrade[i]
                t=[1,i]
                s=0
                for k in range(i,i-ticks,-1):
                    try:
                        diff2=ask_dict[k][orderbook.iloc[i,3]]-ask_dict[k-1][orderbook.iloc[i,3]]+askPriceTrade[i-1]
                    except:
                        break
                    if diff2>threshold2:
                        s=s+diff2
                        t.append(k)
                        #k-(k-1)
                    if s>=diff1:
                        t[0]=0
                        break
                '''if t[0]==0:
                    suspect.append(t)'''
                suspect.append(t)
    return suspect


# In[79]:


def get_vwap(data,multiply):
    vwap=[]
    td=np.asarray(data.iloc[1:,-4])-np.asarray(data.iloc[:-1,-4])
    vd=np.asarray(data.iloc[1:,-3])-np.asarray(data.iloc[:-1,-3])
    for i in range(0,len(td)):
        if vd[i]>0:
            vwap.append(td[i]/(0.0+multiply)/vd[i])
        else:
            vwap.append(vwap[-1])
    return np.array(vwap)

def featureOFI(data,vwap):
    vd=(np.asarray(data.iloc[1:,-3])-np.asarray(data.iloc[:-1,-3]))/2.0
    
    apt=(vwap-np.asarray(data.iloc[:-1,5])+0.0)/(np.asarray(data.iloc[:-1,3])-np.asarray(data.iloc[:-1,5]))
    for i in range(0,len(apt)):
        apt[i]=vd[i]*max(0,min(apt[i],1))
    return list(apt)+[vd[-1]/2.0],list(vd-apt)+[vd[-1]/2.0]


def bid_spoof_cleaning(data,bidPriceTrade,askPriceTrade,threshold,ticks):
    counter=0
    suspect=bid_detect(data,bidPriceTrade,askPriceTrade,threshold,ticks)
    while len(suspect)>1 and counter<10:
        j=0
        temp=0
        counter+=1
        for i in range(0,len(data)):
            if suspect['start'].iloc[j]==i:
                p=data.iloc[i,5]
                data.iloc[i,6]-=threshold
                temp=1
            if temp==1 and i<suspect['end'].iloc[j]:
                for k in range(0,5):
                    if data.iloc[i,4*k+5]==p:
                        data.iloc[i,4*k+6]-=threshold
            if temp==1 and i==suspect['end'].iloc[j]:
                for k in range(0,5):
                    if data.iloc[i,4*k+5]==p:
                        data.iloc[i,4*k+6]-=threshold
                temp=0
                j=min(j+1,len(suspect)-1)
        suspect=bid_detect(data,bidPriceTrade,askPriceTrade,threshold,ticks)
    return data

def ask_spoof_cleaning(data,askPriceTrade,bidPriceTrade,threshold,ticks):
    counter=0
    suspect=ask_detect(data,askPriceTrade,bidPriceTrade,threshold,ticks)
    while len(suspect)>1 and counter<10:
        j=0
        temp=0
        counter+=1
        for i in range(0,len(data)):
            if suspect['start'].iloc[j]==i:
                p=data.iloc[i,3]
                data.iloc[i,4]-=threshold
                temp=1
            if temp==1 and i<suspect['end'].iloc[j]:
                for k in range(0,5):
                    if data.iloc[i,4*k+3]==p:
                        data.iloc[i,4*k+4]-=threshold
            if temp==1 and i==suspect['end'].iloc[j]:
                for k in range(0,5):
                    if data.iloc[i,4*k+3]==p:
                        data.iloc[i,4*k+4]-=threshold
                temp=0
                j=min(j+1,len(suspect)-1)
        suspect=ask_detect(data,askPriceTrade,bidPriceTrade,threshold,ticks)
    return data
