#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 01:23:08 2020
This is the tester function for created models. 
Users can use it to evaluate their models. 


Abdullah BAS 
abdullah.bas@boun.edu.tr
BME Bogazici University
Istanbul / Uskudar
@author: abas
"""

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import exceLoader
from torch.utils.data import Dataset, DataLoader
import config

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

model = load_checkpoint(config.model)
#modelparameters=torch.load('')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
model.eval()


load_test=exceLoader.dataset('FitData_2.xlsx',phase='eval')
test_loader = DataLoader(load_test, batch_size=1,
                                         shuffle=False)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(len(config.classes))]
    n_class_samples = [0 for i in range(len(config.classes))]
    for (age,images, labels) in test_loader:
        
        
        images = images.to(device)

        labels = labels.to(device)
     
        outputs = model(images,age)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        for i in range(len(outputs)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    classes=config.classes
    for i in range(len(classes)):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')


