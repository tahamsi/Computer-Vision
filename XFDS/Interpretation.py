# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:33:53 2021

@author: tahamansouri
"""
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import time

def justification(Cluster,output_layer):
    similar=np.argmin([euclidean(out,output_layer) for out in Cluster['Features']])
    return Cluster.iloc[similar]['Original']

# Kmeans
def create_clusters(k,featuresLastLayer,originals):
    km = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
    y_calss = km.fit_predict(featuresLastLayer)
    allTogether = combine(featuresLastLayer,originals,y_calss,km.cluster_centers_)
    df=pd.DataFrame.from_dict(allTogether)
    df['Rank_First'] = df.groupby('Cluster')['Distance'].rank(method='first')
    return df[df['Rank_First']==1]
    
def combine(inpData,original,Y,centroids):
    res=[]
    for inp, orig,y in zip(inpData,original,Y):
        dic={}
        dic['Original']=orig
        dic['Features']=inp
        dic['Distance']=euclidean(centroids[int(y),:],inp)
        dic['Cluster']=int(y)
        res.append(dic)
    return res

def euclidean(V, U):
    return np.linalg.norm(V-U)

def getRandIndex(ytest , _0_or_1):
    idx=np.random.randint(0,len(ytest))
    while ytest[idx] != _0_or_1:
        idx=np.random.randint(0,len(ytest))
    return idx

def describeInstance(idx,y_pred,y_test,output_layer_new,participating_features,Cluster_Fraud,Cluster_Normal,x_test_for_feature):
    start_time_search = time.time()
    y_true = y_test[idx]
    y_pred=y_pred[idx]
    output_layer_new=output_layer_new[idx]
    if y_pred:
        response=justification(Cluster_Fraud,output_layer_new)
        lable="positive"
    else:
        response=justification(Cluster_Normal,output_layer_new )
        lable="negative"
    
    if y_true:
        true_lable= "positive"
    else: 
        true_lable= "negative"
    
    result=[]
    count=0
    for i,j in zip(response,x_test_for_feature[idx,:]):
        if i==j:
            result.append(str(i)+" == "+str(j))
            count+=1
        else:
            result.append(str(i)+" != "+str(j))
    
    desc  = "The model classified this transaction as "+lable+".\n"
    desc += "While its true class was "+true_lable+".\n"
    desc += "The model has made this decision, because the original patern:\n"
    desc += str(x_test_for_feature[idx,:])
    desc += "\nIs mostly similar to the following: \n"
    desc += str(response)
    desc += "\nWhich is a dominant pattern of the class "+lable+".\n"
    desc += "On the other hand, these two pattern have "+str(count)+" similar points as follow:\n"
    for r in result:
        desc += str(r)
        desc += "\n"
    desc += "the most participating features are as follow: "
    desc += str(participating_features[idx])
    end_time_search = time.time()

    RuntimeSearchValue = end_time_search - start_time_search
    
    return desc , RuntimeSearchValue
