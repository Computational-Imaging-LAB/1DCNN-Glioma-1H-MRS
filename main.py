#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:07:36 2020

This script is for detection IDH and TERTp mutation in gliomas.
1D-CNN is the approach that we use in this study.
This study was submitted to ISMRM 2021


Abdullah BAS 
abdullah.bas@boun.edu.tr
BME Bogazici University
Istanbul / Uskudar
@author: abas
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import optuna
import exceLoader
import config
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from model1DCNN import Net
device=config.device

def get_data_loaders(train_batch_size, test_batch_size):
    """ This function is for initializing data_loaders

    Args:
        train_batch_size (int): Can be manipulated from config.py
        test_batch_size (int): Can be manipulated from config.py

    Returns:
        dataloaders: Returns train and test dataloaders.
    """    
    #load=exceLoaderTERT.dataset('fit2.xlsx',preprocess=False)
    #load_test=exceLoaderTERT.dataset('fit2.xlsx',phase='test',preprocess=False)
    
    load=exceLoader.dataset(config.train_path,responseCol=820,preprocess=True) # defining the dataloaders training
    load_test=exceLoader.dataset(config.test_path,responseCol=820,preprocess=True,phase='test') # defining the dataloaders test
    
    
    train_loader=DataLoader(load,batch_size=train_batch_size,shuffle=True) # creating data_loader from dataset
    test_loader = DataLoader(load_test, batch_size=test_batch_size,
                                         shuffle=False) # creating data_loader from dataset

    return train_loader, test_loader




    
def train(log_interval, model, train_loader, optimizer, epoch):
    """This is for training loop of optuna

    Args:
        log_interval (int): Logging interval
        model (torch model): 1D-CNN model that we created by using model1DCNN.py 
        train_loader (dataloader): train data_loader for training 
        optimizer (torch.optimizer): you can select the optimizer
        epoch (int): Epoch number

    Returns:
        [float]: Returns training loss
    """    

    
    model.train()
    losses=[]
    for batch_idx, (age,data, target) in enumerate(train_loader):
        optimizer.zero_grad() 
        output = model(data.to(device),age.to(device)) # model prediction
        criterion = nn.CrossEntropyLoss() # loss function
        loss=criterion(output, target.to(device))# computing the loss  
        loss.backward() # backpropogation
        optimizer.step() # optimize
        losses.append(loss.item()) # listing loss
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))  
    return losses # returning the losses

    
def test(model, test_loader):
    """ Testing loop for optuna

    Args:
        model (torch.model): model to input
        test_loader (dataloader): dataloader for testing 

    Returns:
        accuracy [float]: accuracy
        acc1 [float]: accuracy of class1
        acc2 [float]: accuracy of class2
        losses [list]: loss list
    """
    
    model.eval()
    test_loss = 0 # initializing the required variables
    correct = 0
    num_classes=[0,0]
    num_samples=[0,0]
    losses=[]

    with torch.no_grad():
        for age, data, target in test_loader:
            output = model(data.to(device),age.to(device))
            criterion = nn.CrossEntropyLoss()
            test_loss=criterion(output, target.to(device)) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            losses.append(test_loss)
            for i,_ in enumerate(pred):
                if pred[i]==target[i]:
                    num_classes[target[i]]+=1
                num_samples[target[i]]+=1
            
            
            correct += pred.eq(target.to(device).view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset) # computing the accuracy
    acc1=100.*num_classes[0]/num_samples[0] # computing the class-wise accuracies
    acc2=100.*num_classes[1]/num_samples[1] 
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(f'Acc1:{acc1}/{num_samples[0]} , Acc2:{acc2}/{num_samples[1]}')
    
    return accuracy,acc1,acc2,losses
 




def train_optuna(trial):
    """This function is for optimizing the hyperparameters. 

    Args:
        trial (optuna.study): This comes from optuna. Initialized study

    Returns:
        test_loss [float]: Test loss value. In this study optuna tries to minimize this value.
    """    
    global best_booster # to find best model to save
    
    # creating the optimization parameters.
    # manipulation can be done with changing cfg parameters
    cfg = { 'device' : config.device,
            'train_batch_size' : config.train_batch_size,
            'test_batch_size' : config.test_batch_size,
            'n_epochs' :trial.suggest_categorical('epochs',[50,60,100,500]),
            'seed' : config.seed,
            'log_interval' : 1,
            'save_model' : False,
            'lr' : trial.suggest_loguniform('lr', 1e-5, 1e-2),          
            'momentum': trial.suggest_uniform('momentum', 0.1, 0.99),
            'optimizer': trial.suggest_categorical('optimizer',[optim.SGD, optim.RMSprop]),
            'activation': trial.suggest_categorical('activation',[F.relu,F.sigmoid,F.leaky_relu])}
  
    torch.manual_seed(cfg['seed']) # for reproducibility
    train_loader, test_loader = get_data_loaders(cfg['train_batch_size'], cfg['test_batch_size'])
    model = Net(cfg['activation']).to(device)  # model defining
    optimizer = cfg['optimizer'](model.parameters(), lr=cfg['lr']) # optimizer defining
    losses_train=[]
    losses_test=[]
    accuracies=[]
    for epoch in range(1, cfg['n_epochs'] + 1):
        loss_train=train(cfg['log_interval'], model, train_loader, optimizer, epoch) # function train
        test_accuracy,acc1,acc2,test_loss = test(model, test_loader) # function test
        losses_train.append(loss_train) 
        losses_test.append(test_loss)
        tester=test_accuracy
        accuracies.append(test_accuracy)
        if test_loss[0]<best_booster:
          
          # checkpoint is for loading and saving the hyperparameters
          checkpoint = {'model': Net(cfg['activation']),
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epochs':epoch,
                    'activation':cfg['activation'],
                    'lr':cfg['lr'],
                    'batch_size':cfg['train_batch_size'],
                    'acc1': acc1,
                    'acc2':acc2,
                    'optimizerName':cfg['optimizer'],
                    'test_loss':losses_test,
                    'train_loss':losses_train,
                    'accuracies': accuracies
                     }
          
          torch.save(checkpoint, f'{config.output}/PRTERT_Mut_{test_accuracy}.pth') # saving the model parameters
    
          best_booster=test_loss[0]
    return test_loss[0]
best_booster=110
    
    
    



if __name__ == '__main__':

  sampler = optuna.samplers.TPESampler()
      
  study = optuna.create_study(sampler=sampler, direction='minimize') # loss value is the value that we want to minimize
  study.optimize(func=train_optuna, n_trials=50)


  df=study.trials_dataframe() # converting study to dataframe 
  study.best_trial
  optuna.visualization.plot_parallel_coordinate(study,params=['lr','momentum']) # plotting the parameters using paralel plot
















   
