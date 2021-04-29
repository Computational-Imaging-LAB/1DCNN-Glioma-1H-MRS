#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 10:30:03 2020
This script is for generating synthetic data using trklearn.py
Optional.

Abdullah BAS 
abdullah.bas@boun.edu.tr
BME Bogazici University
Istanbul / Uskudar
@author: abas
"""



import pandas as pd
import numpy as np
from trklearn import ADASYN,SMOTE

df=pd.read_excel('FitData_2.xlsx')


df=df.iloc[85::,:]

inds=(df.iloc[:-1,8].isna())
inds=np.where(inds==False)
df=df.iloc[inds[0],:]

dfNp=np.array((df.iloc[:,13::]))  
dfNp=np.insert(dfNp,0,np.array(df.iloc[:,1]),axis=1)


out,response=ADASYN.fit_resample(dfNp,np.array((df.iloc[:,10])),threshold=0.9)
out2,response2=SMOTE.fit_resample(dfNp,np.array((df.iloc[:,10])),threshold=0.9)

an_array = np.insert(out, 820, response, axis=1)

df=pd.DataFrame(an_array)
dfS=df.sort_values(by=df.columns[150])

with pd.ExcelWriter('fit22.xlsx') as writer:
    dfS.to_excel(writer)
    
    
