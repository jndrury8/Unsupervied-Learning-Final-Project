#!/usr/bin/env python
# coding: utf-8

# # Penguin Species Classification Final Project

# The goal of this project is to be able to correctly classify the species of penguin given measurements about the penguin. 

# The dataset contains measurement of features of penguins and what species of penguin they are. The goal of this is to try and develope a clustering model that can accurately predict what sort of penguin species each data point is.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.cluster.hierarchy as sch
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore


# ## Importing the data

# Importing the data using its URL and then subsequently cleaning it, making it easier to interpret, removing unnecessary columns.

# In[2]:


data=pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/5489301/9095883/penguins.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240819%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240819T190752Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=76dbe9ecebfc581d314161c9dc81f083d94af3184b721f08f0c9ab4ba79086799e502bec69ed532455cee5db5e28242eafd38009a27bcf2693c219dba1f70a70def5694b274f1c7f4fdcf030afb1d9808bf2724848d3116e1f06a5fc601c998e18e194bfdbdc41dd0a44cc250dfce01f5946a3406eae5528aaf8c13d2b493e4a41490c4d446330a706ce0469432fef24b6cb19b5096d9af88af663e09b9b840097baa92152930625eeb0ff22e146e30272975922d38aa4b410799073f13e251f557ed8a0e72a7ce71b46a205014f80934897c9835a66f8705dd6f3ba58dc9b19eb479b7bcac4ffa8475eef0025bcea70de0fc7340ef172b10da05dd86d883ef7')
data=data.dropna()
labels=data['species'].values
data1=data.drop(['id', 'island','year','sex','species'], axis=1)


# ## EDA

# Finding datatypes and shapes of the data we are working with and removing outliers (outlier detection code removed because no outliers were detected)  

# In[3]:


data.info()
data1.info()
print(labels.shape)
speclabels= list(np.unique(labels))
print(speclabels)
data2=data1.values

Originally the data had 344 rows, though some had to be removed do to NaN values
# In[4]:


df = pd.DataFrame(data2)
print(df.describe())


# ## Agglomerative Clustering:

# ## Pre Standardization

# Using inspiration from Week 2, we check the accuracy on the clustering model for 3 clusters with the default parameters

# In[5]:


def label_permute_compare(lab,modlab,n=3):
    numLabels=list(set(modlab))
    permute=itertools.permutations(numLabels,3)
    speclabels= list(np.unique(lab))
    BestAcc=0
    mapping = {trueval: i for i, trueval in enumerate(speclabels)}
    labmap= [mapping[trueval] for trueval in lab]
    for i in permute:
         NewLabels =[i[x] for x in modlab]
         acc = accuracy_score(labmap,NewLabels)
         if acc > BestAcc:
            BestAcc= acc
            bestcm= confusion_matrix(labmap, NewLabels)
    print(bestcm)
    print(BestAcc)
    return BestAcc
model= AgglomerativeClustering(n_clusters=3).fit(data2)
label_permute_compare(labels,model.labels_)


# This accuracy is not ideal. We will do something similar from Week 2 where we found the ideal parameters.

# In[6]:


lm = ['ward', 'complete', 'average', 'single']
dm = ['euclidean', 'manhattan', 'cosine', 'l1', 'l2']
bestacc=0
bestmetric=None
bestlinkage=None
for i in lm:
    for j in dm:
        if i == 'ward' and j != 'euclidean':
            continue
        model2= AgglomerativeClustering(n_clusters=3, affinity= j, linkage=i).fit(data2)
        acc = label_permute_compare(labels, model2.labels_)
        if  acc > bestacc:
            bestacc=acc
            bestlinkage=i
            bestmetric=j
print("Best Combination",bestlinkage,bestmetric,"Accuracy",bestacc)
print('Confusion Matrix of best Combination:')
model= AgglomerativeClustering(n_clusters=3, affinity= bestmetric, linkage=bestlinkage).fit(data2)    
label_permute_compare(labels, model.labels_)    


# This accuracy is still not ideal, only .02 higher than the default parameters. Lets see if standardizing the data improves the accuracy of the model. We see that the best linkage is 'average' and the best metric is 'cosine'. The confusion matrix shows 109 errors, which is extremely high for a data set that only contains 333 data points.
# 

# #### Dendrogram of 3 Levels

# In[7]:


LinkMat = sch.linkage(data2, method='average')
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(LinkMat,truncate_mode='level',p=3)
plt.show()


# ## Standardizing the Data

# In[8]:


stanscaler=StandardScaler()
StandData=stanscaler.fit_transform(data2)
model= AgglomerativeClustering(n_clusters=3, affinity= bestmetric, linkage=bestlinkage).fit(StandData)    
label_permute_compare(labels, model.labels_) 


# Very significant improvement! Standardization can help with model accuracy and efficiency. As we can see here, it greatly helped the effectiveness of our model. Only 6 errors.

# In[ ]:





# ## K-Means Clustering

# Lets perform K-Means clustering on unstandardized data

# In[10]:


kmeans = KMeans(n_clusters=3, random_state=7)
kmeans.fit(data2)
acc=label_permute_compare(labels,kmeans.labels_,n=3)
print(acc)


# Not great, worse than our original model before it was standardized. Lets see how it performs if we use the standardized data.

# In[11]:


kmeans = KMeans(n_clusters=3, random_state=7)
kmeans.fit(StandData)
acc=label_permute_compare(labels,kmeans.labels_,n=3)
print(acc)


# Significant improvement! Not as good as our final Agglomerative Clustering model using standardized data

# ## Conclusion

# In this project, we were able to see two different modeling techniques and how well they performed with the given data set. Initially, neither of them performed well at all. Their performance time was good, but they were very inaccurate. After we standardized the data, the accuracy of both of the models increased dramatically. It can be said that the Agglomerative Clustering method can predict the species of penguin with great accuracy, only producing 6 misclassifications. The K-Means model also became accurate after standardization, though not to the same level as our  Agglomerative Clustering model.

# In[ ]:




