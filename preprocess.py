#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 01:51:16 2020
This is the script for preprocessing of MRS spectrums. 

Users can switch between different methods using 
the method parameter of functions


Abdullah BAS 
abdullah.bas@boun.edu.tr
BME Bogazici University
Istanbul / Uskudar
@author: abas
"""

from sklearn import preprocessing
from scipy.signal import savgol_filter

def scaler(X,method=2):
    
    
    if method==0:
        X_scaled=preprocessing.scale(X)
    elif method==1:
        scaler=preprocessing.StandardScaler().fit(X)
        X_scaled=scaler.transform(X)
    elif method==2:
        scaler=preprocessing.MinMaxScaler()
        X_scaled=scaler.fit_transform(X)
        
    elif method==3:
        scaler=preprocessing.RobustScaler().fit(X)
        X_scaled=scaler.fit_transform(X)
    else:
        X_scaled=X

    return X_scaled,scaler


def transformation(X,method=1,powerMet='yeo-johnson'):
    
    if method==0:
        scaler=preprocessing.QuantileTransformer(random_state=0)
        X_tr=scaler.fit_transform(X)
    elif method==1:
        scaler=preprocessing.PowerTransformer(method=powerMet,standardize=False)
        X_tr=scaler.fit_transform(X)
    else:
        X_tr=X
    return X_tr,scaler


def normalizer(X,norm='l2'):
    
    
    X_norm=preprocessing.normalize(X,norm=norm)
    
    
    return X_norm

def smoother(X,window=11,order=2):
    
    
    X_smooth=savgol_filter(X,window,order)
    
    return X_smooth

# if __name__=='__main__'
#     import pandas as pd
#     import numpy as np
#     a=pd.read_excel('FitData.xlsx')

#     arr=np.array(a.iloc[:,:-1])
#     a_sm=smoother(arr[0])
#     a_nm=normalizer(arr)
#     a_tm=transformation(arr)
#     a_sc=scaler(arr)


 