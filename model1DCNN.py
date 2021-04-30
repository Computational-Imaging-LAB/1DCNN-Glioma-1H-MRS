#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:23:32 2021

1D-CNN Model

Abdullah BAS 
abdullah.bas@boun.edu.tr
BME Bogazici University
Istanbul / Uskudar
@author: abas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,activation):
        """Initializing the 1D-CNN 

        Args:
            activation (nn.Functional): Activation function comes 
            from optuna optimization
        """        
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 14, 40)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(14, 7, 40)
        self.bn1=nn.BatchNorm1d(14)
        self.bn0=nn.BatchNorm1d(1)
        self.dropout=nn.Dropout(p=0.2)
        self.act=(activation)
        self.conv3 = nn.Conv1d(7, 7, 40,padding=0)
        self.fc1 = nn.Linear(7*68, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(11, 2)


    def forward(self, x,age):
        """Forward blocks of the model. Input data moves through from this function.

        Args:
            x (float): input MRS spectrum
            age (float): Age of patient

        Returns:
            [float]: output of the model
        """        
        x = self.pool(self.act(self.conv1(x)))  # -> n, 6, 14, 14
        x=self.dropout(x)
        x=self.bn1(x)
        x = self.pool(self.act(self.conv2(x)))  # -> n, 16, 5, 5
        x = self.pool(self.act(self.conv3(x)))  # -> n, 16, 5, 5      
        x=self.dropout(x)     
        x = x.view(-1, x.shape[1] *x.shape[2])        
        x = self.act(self.fc1(x)) 
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        age=(age).unsqueeze(1)/100
        x=torch.cat((x,age),dim=1)
        x = self.fc4(x)     # -> n, 10
                     
        return x
