#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Apr 29 18:19:32 2021

Config file for the project

Abdullah BAS 
abdullah.bas@boun.edu.tr
BME Bogazici University
Istanbul / Uskudar
@author: abas
"""

import torch
train_path='Data/train/data.xlsx'

test_path='Data/test/data.xlsx'

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_batch_size = 64
test_batch_size=20
seed=0

output='Denemeler/'
