#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 01:15:09 2020

@author: abas
"""

import pandas as pd
import numpy as np



data=pd.read_excel('FitData.xlsx')
array=np.array(data.iloc[0:205,:])

clas0=array[array[:,820]==0,:-1]
clas1=array[array[:,820]==1,:-1]




def crossing(clas,cl):
    
    samples=clas.shape[0]
    lista=[]
    for k in range(1,1000):
        out=[]
        for i in range (0,11):
            print(i)
            
            idx=np.random.randint(0,samples-1)
            print(idx)
            new=clas[idx,i*82:(i+1)*82]
            out=np.append(out,new)
            
        out=np.append(out,cl)
       
        lista.append(np.transpose(out))
        
        
    return np.array(lista)
        
      
out1=crossing(clas1,1)
out0=crossing(clas0,0)

outs=np.vstack((out0,out1))
df=pd.DataFrame(outs,columns=(data.columns))

with pd.ExcelWriter('outputs.xlsx') as writer:
    df.to_excel(writer)