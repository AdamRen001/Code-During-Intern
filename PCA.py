
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
files = sorted(os.listdir('/data/public/cleandata/featurematrix/'))
file_paths =['/data/public/cleandata/featurematrix/'+f for f in files]
data = pd.read_csv(file_paths[0])
for f in file_paths[1:len(file_paths)]:
    df = pd.read_csv(f)
    data = pd.concat([data,df])
origColumns = df.columns
selectColumns = []
for col in origColumns:
	if ((col.find('000016')<0) and (col.find('000300')<0)
    and (col.find('000905')<0) and (col.find('510050')<0)
    and (col.find('510300')<0) and (col.find('510500')<0)
    and (col.find('IF')<0)): 
        #"000016","000300","000905","510050","510300","510500"
		selectColumns.append(col)

dfselect = data[selectColumns]
midmatrix = dfselect.iloc[:,3:]
rtnmatrix = midmatrix.copy(deep=True)
for name in rtnmatrix.columns:
    rtnmatrix[name] = rtnmatrix[name] - rtnmatrix[name].shift(120)
print (rtnmatrix.head())


# In[3]:
new_rtnmatrix = rtnmatrix.iloc[120:,]
# In[12]:


#pca on rtnmatrix
from sklearn.decomposition import PCA
pca = PCA(svd_solver='auto')
pca.fit(new_rtnmatrix)


# In[25]:


components = pca.components_


# In[27]:


pc1 = components[0]
pc2 = components[1]
pc3 = components[2]
pc4 = components[3]


# In[ ]:


pc1.to_csv('pc1_weights.csv',index=None)
pc2.to_csv('pc2_weights.csv',index=None)
pc3.to_csv('pc3_weights.csv',index=None)
pc4.to_csv('pc4_weights.csv',index=None)

