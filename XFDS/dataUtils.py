# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:13:35 2021

@author: tahamansouri
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def load_data(train,test):
    """
    This function receives the name of train and test datasets and returnsthe following:
    datasetTrain
    x_train_for_feature
    datasetTest
    x_test_for_feature
    """
    datasetTrain = pd.read_csv('dataset/'+train)
    datasetTrain= datasetTrain.dropna(how='all')
    datasetTrain['flag_fraude_cat'] = datasetTrain['flag_fraude_cat'].replace(['S'],1)
    datasetTrain['flag_fraude_cat'] = datasetTrain['flag_fraude_cat'].replace(['N'],0)
    datasetTrain = datasetTrain.astype(int)
    x_train_for_feature = datasetTrain.iloc[:, 0:17].values
    
    datasetTest = pd.read_csv('dataset/'+test)
    datasetTest= datasetTest.dropna(how='all')
    
    datasetTest['flag_fraude_cat'] = datasetTest['flag_fraude_cat'].replace(['S'],1)
    datasetTest['flag_fraude_cat'] = datasetTest['flag_fraude_cat'].replace(['N'],0)
    datasetTest = datasetTest.astype(int)
    x_test_for_feature = datasetTest.iloc[:, 0:17].values
    
    return datasetTrain, x_train_for_feature,datasetTest,x_test_for_feature

def preprocessing(datasetTrain,datasetTest):
    # Scaling and One hot encoding
    columns_to_oneHotEncode = ["mcc_cat","mcc_ant_cat","cep_cat","valor_trans_ant_cat","lim_cred_cat","bandeira_cat","score_cat","tp_pessoa_cat","dif_score_cat"]
    columns_to_scale = ["cep_ant_cat","valor_trans_cat","pos_entry","variante_cat","trans_nacional_cat","qtde_parc_cat","velocidade_cat","trans_limit_cat"]
    columns_classLabel = ["flag_fraude_cat"]
    df_Class = datasetTrain[columns_classLabel]
    df_OneHot = pd.get_dummies(datasetTrain, columns=columns_to_oneHotEncode)
    for i in df_OneHot.columns:
        v = (i[i.rfind("_")+1:])
        if v =='1':
            df_OneHot = df_OneHot.drop(columns=[i])
    df_Scale = datasetTrain.loc[:,columns_to_scale]
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    df_Scale.loc[:,columns_to_scale] = min_max_scaler.fit_transform(df_Scale)
    datasetTrainTransformed = pd.concat([df_OneHot, df_Scale,df_Class], axis=1)
    X_train = datasetTrainTransformed.iloc[:,:-1].values
    y_train = datasetTrainTransformed.iloc[:, -1].values
    
    df_ClassTest = datasetTest.loc[:,columns_classLabel]
    df_OneHotTest = pd.get_dummies(datasetTest, columns=columns_to_oneHotEncode)
    
    for i in df_OneHotTest.columns:
        v = (i[i.rfind("_")+1:])
        if v =='1':
            df_OneHotTest = df_OneHotTest.drop(columns=[i])
    df_ScaleTest = datasetTest.loc[:,columns_to_scale]
    df_ScaleTest.loc[:,columns_to_scale] = min_max_scaler.fit_transform(df_ScaleTest)
    datasetTestTransformed = pd.concat([df_OneHotTest, df_ScaleTest,df_ClassTest], axis=1)
    X_test = datasetTestTransformed.iloc[:, 0:-1].values
    y_test = datasetTestTransformed.iloc[:, -1].values
    
    return X_train,y_train,X_test,y_test
