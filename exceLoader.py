#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:40:41 2020
Abdullah BAS
abdullah.bas@boun.edu.tr
BME Bogazici University
Istanbul / Uskudar
@author: abas
"""

import numpy as np

import pytorch_lightning as pl
import pickle
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from preprocess import transformation,scaler,normalizer,smoother
import pandas as pd
from preprocess import scaler as scl
from scipy.fft import fft,ifft 
class dataset(Dataset):
    
    """ This function initializes the dataset. 

        Args:
            path (string): Input path
            gtpath (string,optional): Path for ground truths. Default to None 
            responseCol (string,optional): If ınput dataset has the response in it you can simply assign it to this value. Note: 0 for first,  1 for second column same as python. Take that into account
            phase (str, optional): Defaults to 'train'.
            preprocess (bool, optional): Switch for preprocess. Defaults to True.
            smooth (bool, optional): Switch for smoothing. Defaults to True.
            normalise (bool, optional): Switch for normalise. Defaults to True.
            transform (bool, optional): Switch for yeo-johnson power transformation. Defaults to True.
        """   
    def __init__(self,path,gtpath=None,responseCol=-1,phase='train',preprocess=True,smooth=True,normalise=True,transform=True):
             
        self.normalise=normalise
        self.exc=pd.read_excel(path)
        self.phase=phase
        self.smooth=smooth
        self.normalise=normalise
        self.transform=transform
        self.preprocess=preprocess
        if gtpath is not None:
            self.response=np.load(gtpath)
        else:
            self.response=np.array(self.exc.iloc[:,responseCol])
            self.excarr=np.array(self.exc.drop(self.exc.columns[[responseCol]],axis=1))
        
        
        if phase=='train':
            self.excarr=np.array(self.exc)
            
            if self.preprocess:
                #self.excarr=normalizer(self.excarr)
                self.excarr=smoother(self.excarr)
                self.excarr,self.scale=scl(self.excarr)
                self.excarr,self.transformater=transformation(self.excarr)
                if not os.path.isdir('preprocess'):
                    os.mkdir('preprocess')
                pickle.dump(self.scale, open('preprocess/scaler.pkl', 'wb'))
                pickle.dump(self.transformater, open('preprocess/transformater.pkl', 'wb'))
                
        elif phase=='valid':
            #self.excarr=normalizer(self.excarr)
            self.excarr=smoother(self.excarr)
            scaler=np.load('preprocess/scaler.pkl',allow_pickle=True)
            transformater=np.load('preprocess/transformater.pkl',allow_pickle=True)
            self.excarr=scaler(self.excarr)
            self.excarr=transformater(self.excarr)
            
    def __len__(self):
        return len(self.excarr)
    
    
    
    def __getitem__(self,idx=None):
        
        spectrum=self.excarr[idx,:]
        response=self.response[idx]
        age=torch.tensor(self.exc.iloc[idx,1]).type(torch.float32)
        
        return age,torch.tensor(spectrum).type(torch.float32).unsqueeze(0),torch.tensor(response).type(torch.long)
