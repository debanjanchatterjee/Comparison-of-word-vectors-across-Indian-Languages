#!/usr/bin/env python
# coding: utf-8

# In[19]:


import csv
import pandas as pd
import os


# In[20]:


def calculateCosineSimilarity(d1, d2):
    cos_sim = np.dot(d1, d2)/(np.linalg.norm(d1)*np.linalg.norm(d2))
    return cos_sim


# In[21]:


fin='language.csv'
dataset=[]
upd_dataset=[]
cnt=0
with open(fin,errors='ignore') as file:
    reader = csv.reader(file)
    data = list(reader)
#     print(data)
#     dataset.append(data)
#     cnt+=1
#     if cnt>=1:
#         break
# #     dataset.append(data)
    
print(len(data))

wals_lan=['ben','hin','tel','tml','guj','mhi','pan','mym','oya','knd']
lan_dict=dict()
for row in data:
    wals_code=row[0]
    if wals_code in wals_lan:
        cur_features=list()
        for i in range(10,len(row)):
            l=row[i].split()
            if len(l)>1:
                cur_features.append(int(l[0]))
            else:
                cur_features.append(int(0))

                  
                                    
    
                
        lan_dict[wals_code]=cur_features
             
# print(lan_dict)            


# In[22]:


fin='indian_languages.csv'
ind_data=[]
with open(fin,errors='ignore') as file:
    reader = csv.reader(file)
    ind_data=list(reader)

ind_lan_dict=dict()   
for row in ind_data:
    ind_lan_dict[row[3]]=row[0]
print(ind_lan_dict)
    


# In[23]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

dataset=[]
lan_names=[]
lan_codes=[]
for lan in lan_dict:
    lan_codes.append(lan)
    lan_names.append(ind_lan_dict[lan])
    fv=lan_dict[lan]
    dataset.append(fv)

dataset_array=np.array(dataset)

# print(dataset_array)
# scaler = StandardScaler()

# scaler.fit(dataset_array)
# print(scaler.mean_)
# dataset_array=scaler.transform(dataset_array)
# print(dataset_array)



# print(dataset_array)
# scaler = MinMaxScaler()

# scaler.fit(dataset_array)
# print(scaler.data_max_)
# dataset_array=scaler.transform(dataset_array)
# print(dataset_array)


# dataset_array = preprocessing.scale(dataset_array)
# print(dataset_array)
# print('Bengali -Hindi',calculateCosineSimilarity(dataset_array[0],dataset_array[1]))
# print('Bengali-  Telegu',calculateCosineSimilarity(dataset_array[0],dataset_array[2]))
# print('Hindi - Telegu',calculateCosineSimilarity(dataset_array[1],dataset_array[2]))

for i in range(0,len(dataset_array)):
    for j in range(0,len(dataset_array)):
        if i<j:
            print('Cosine similarity between',lan_codes[i],'and',lan_codes[j],'-',round(calculateCosineSimilarity(dataset_array[i],dataset_array[j]),2))
                


# In[24]:


#Import KMeans module
from sklearn.cluster import KMeans
 
#Initialize the class object
kmeans = KMeans(n_clusters= 3)
 
#predict the labels of clusters.
label = kmeans.fit_predict(dataset_array)
print(label)
print(lan_codes)


# In[25]:


from sklearn.cluster import AgglomerativeClustering

for i in range(2,8):
    clustering = AgglomerativeClustering(n_clusters= i).fit(dataset_array)
    print(clustering.labels_)


# In[26]:


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# In[27]:


plt.rcParams["figure.figsize"] = (15,15)
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(dataset_array)
print(model.labels_)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=4, labels=lan_names)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# In[28]:


model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(dataset_array)
print(model.labels_)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=4, labels=lan_names)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# In[ ]:





# In[ ]:




