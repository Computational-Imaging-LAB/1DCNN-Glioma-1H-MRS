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
    """In this function users can select a scaler among sklearn.preprocessing.scale,StandardScaler,MinMaxScaler,RobustScaler

    Args:
        X (float): Input array. 
        method (int, optional): It is for choosing the scaler.0:scale,1:StandardScaler,2:MinMaxScaler,3:RobustScaler. Defaults to 2.

    Returns:
        [float]: X_scaled scaled version of X 
        [scaler]: Scaler with parameters estimated from input X.
    """    
    
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
    """Power transformation

    Args:
        X (float): Input data
        method (int, optional): 0:QuantileTransformer,1:Power Transformer. Defaults to 1.
        powerMet (str, optional): It is essential for method 1. Not included in method 0. Defaults to 'yeo-johnson'.

    Returns:
        [float]: X_tr transformed version of X.
        [transformer]: transformer with the estimated values from the input X
    """    
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
    """Normalizer

    Args:
        X (float): Input data
        norm (str, optional): You can choose between L1 and L2. Defaults to 'l2'.

    Returns:
        [float]: X_norm is the normalized output of input X
    """    
    
    X_norm=preprocessing.normalize(X,norm=norm)
    
    
    return X_norm

def smoother(X,window=11,order=2):
    """Savitzky-Golay filtering

    Args:
        X (float): Input data
        window (int, optional): Smoothin kernel length. Defaults to 11.
        order (int, optional): Filter order. Defaults to 2.

    Returns:
        [float]: X_smooth is the smoothed output of input X.
    """    
    
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


 