# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:17:41 2020

This script is for generating synthetic data. 
You can use multi-class data to generate balance dataset.

Abdullah BAS 
abdullah.bas@boun.edu.tr
BME Bogazici University
Istanbul / Uskudar
@author: abas
"""
import numpy as np
from sklearn import neighbors
from sklearn.cluster import KMeans




def EuclidianDistance(data1,data2):
    
    dist=np.sqrt(sum([(np.square(x)) for x in np.array(data1)-np.array(data2)]))

    return dist     
    
    
    
    
    
    
class MinMaxNormalization():

    """
    Min-Max Normalization.  Was using in conjunction of ADASYN to test results
        data:  Data to be normalized
        axis:  0 is by columns, 1 is by rows
     returns:  Normalized data
    """
    def __init__(self, data, axis=0):
        self.row_min = np.min(data, axis=axis)
        self.row_max = np.max(data, axis=axis)
        self.denominator = abs(self.row_max - self.row_min)

        # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
        for index, value in enumerate(self.denominator):
            if value == 0:
                self.denominator[index] = 1

    def __call__(self, data):
        return np.divide((data - self.row_min), self.denominator)




#☻target=np.array(X.iloc[:,-1])
class ASUWO():
  
    
    
    def ASUWO(Xnp,target,n,k,irt,knn,de=-1,normalization=1):
        """ASUWO is supporting multi-class synthetic data generation. 

        Args:
            Xnp (type:all): Input array must be numpy array
            target (type:all): Corresponding response to Xnp
            n ([type]): 
            k (int): Neighbours number
            irt ([type]): [description]
            knn ([type]): [description]
            de (int, optional): [description]. Defaults to -1.
            normalization (bool, optional): Switch for normalization. Defaults to 1.
        """        
        
        targetarry=np.unique(np.array(target))
        targetNums=[]
        targetClasses=[]
        for classes in targetarry:
            targetNums.append(sum(target==classes))
        
        mi=max(targetNums)
        ms=min(targetNums)
        maxClassİnd=targetNums.index(mi)
        tn=list(targetNums)
        del tn[tn.index(mi)]
        
        def semiUnsCls(Xnp,target,normalization=normalization):
            """[summary]

            Args:
                Xnp ([type]): [description]
                target ([type]): [description]
                normalization ([type], optional): [description]. Defaults to normalization.
            """                                       
            test=np.Inf
            clf = neighbors.KNeighborsClassifier()
            clf.fit(Xnp[target==targetarry[targetNums.index(mi)]], target[target==targetarry[targetNums.index(mi)]])
            clf2 = neighbors.KNeighborsClassifier()
            clf2.fit(Xnp, target)
            Xsub=np.array(Xnp)
            Xsub=Xnp[target==targetarry[targetNums.index(mi)]]
            Xsub2=list(Xsub)
            cond=0
            indices2=[]
            while len(Xsub2)>0:
                
                xi=Xsub[cond]
                cond=cond+1
                neighbours = clf.kneighbors(xi.reshape(1,-1), n_neighbors=2, return_distance=True)
                
                neig2=clf2.kneighbors(xi.reshape(1,-1), n_neighbors=2, return_distance=True)
                neig3=clf2.kneighbors(Xsub[neighbours[1][0][1] ].reshape(1,-1),n_neighbors=2)
                print( "{} değerli  {} elemanın {} elemana en yakın".format(neighbours[0][0][1],neig2[1][0][0],neig3[1][0][1])   )
                if  neighbours[0][0][1]<test and neig2[1][0][0]==neig3[1][0][1]:
                    test=neighbours[0][0][1]
                    indices=[cond,int(neighbours[1][0][1])]
                    del Xsub2[indices[0]]
                    del Xsub2[indices[1]]
                indices2.append(indices)
                indices=[]
    
        
        filteredClusters=[]
        kmeans = KMeans(n_clusters=k, random_state=0).fit(Xnp)
        clusters=kmeans.predict(Xnp)
        
      
        targetarry=np.unique(np.array(target))
        targetNums=[]
        targetClasses=[]
        for classes in targetarry:
            targetNums.append(sum(target==classes))
            
        mi=max(targetNums)
        maxClassİnd=targetNums.index(mi)
        tn=list(targetNums)
        del tn[tn.index(mi)]  
        
        
        for i in range(k):
            buff=(np.multiply((clusters==i),target+1))
            for j in targetarry:
                if irt<((sum(buff==j+1)+1)/(sum(buff==targetarry[targetNums.index(mi)]+1)+1)):
                    filteredClusters.append(i)

class ADASYN():
    
  def ADASYN(Xnp,target,verbose=True,B=1,K=15,threshold=0.7):
      """[summary]

      Args:
          Xnp (np.array): Input array must be numpy
          target (np.array): response array
          verbose (bool, optional): If zero will not output any verbose. Defaults to True.
          B (int, optional): B is the balance ratio that you want to reach. Defaults to 1.
          K (int, optional): Numbers of neighbours. Defaults to 15.
          threshold (float, optional): Activation threshold. Above this balance ratio function will not work. Defaults to 0.7.

      Returns:
          [type]: [description]
      """
    clf = neighbors.KNeighborsClassifier()
    clf.fit(Xnp, target)
    
    
    
    
    targetarry=np.unique(np.array(target))
    targetNums=[]
    targetClasses=[]
    for classes in targetarry:
        targetNums.append(sum(target==classes))
        
    mi=max(targetNums)
    maxClassİnd=targetNums.index(mi)
    tn=list(targetNums)
    del tn[tn.index(mi)]

    r=[]
    r2=[]
    riP=[]
    riP2=[]
    si=[]
    si2=np.zeros((1,Xnp.shape[1]+1))
    for idx,ms in enumerate(tn):
    
       d=ms/mi
       if d<threshold:
           
        G=(mi-ms)*B
        msClass=Xnp[target==targetNums.index(ms)]
        r2=[]
        
        for idx2,xi in enumerate(msClass):
            
            neighbours = clf.kneighbors(xi.reshape(1,-1), n_neighbors=K, return_distance=False)[0]
            delta=sum(target[neighbours]==idx)
            r2.append((K-delta)/K)
            if verbose==True:
                print("Class {}   X{}     R{}={}".format(idx,idx2,idx2,(K-delta)/K))
        
        r.append(r2)
        for idx3,ri in enumerate(r2):
            
            riP2.append(ri/sum(r2))
            if verbose==True:
                print("riP{} have created = {}".format(idx3,ri/sum(r2)))
        riP.append(riP2)

        gi=np.multiply(riP2,G)
        riP2=[]
        gi=np.round(gi)
        for idx4,num in enumerate(gi):
          for idx5 in range(int(num)):
            neighbours = clf.kneighbors(msClass[idx4,:].reshape(1,-1), n_neighbors=K, return_distance=False)[0]
            buff=(msClass[idx4,:].reshape(1,-1)+np.multiply((Xnp[neighbours[np.random.randint(0,K)]]-
                       msClass[idx4,:].reshape(1,-1)),np.random.random(1)))
            si2=np.vstack((si2,np.array(list(buff[0])+[(targetarry[targetNums.index(ms)])])))
            if verbose==True:
                print("{}.element of {}gi created which is the {}.element of created data ".format(idx5+1,idx4,idx4+idx5))
        si.append(si2[1:-1,:])
        r2=[]
        si2=np.zeros((1,Xnp.shape[1]+1))
    return si,riP,r,gi
    


  def fit_resample(Xnp,target,B=1,K=15,threshold=0.7):
        
        si,riP,r,gi=ADASYN.ADASYN(Xnp,target,B=B,K=K,threshold=threshold)
        for x in si:
            Xnp=np.vstack((Xnp,x[:,:-1]))
            target=np.array(list(target)+list(x[:,-1]))
        return Xnp,target
    
    
class SMOTE():
    
    
  def SMOTE(Xnp,target,N=-1,threshold=0.7,verbose=True):
        
        
    targetarry=np.unique(np.array(target))
    targetNums=[]
    targetClasses=[]
    for classes in targetarry:
        targetNums.append(sum(target==classes))
    
    mi=max(targetNums)
    maxClassİnd=targetNums.index(mi)
    tn=list(targetNums)
    del tn[tn.index(mi)]
    
    
    
    clf = neighbors.KNeighborsClassifier()
    clf.fit(Xnp, target)
    
    
    xiP2=[]
    xiP=np.zeros((1,Xnp.shape[1]+1))
    for idx,Ins in enumerate(tn):
        if N==-1:
            N=np.ceil(mi/Ins).astype(int)
            N=N-1
        Xnsp=Xnp[target== targetarry[targetNums.index(Ins)],:]
        clf = neighbors.KNeighborsClassifier()
        clf.fit(Xnsp, target[target==targetarry[targetNums.index(Ins)]])
        if verbose==True:
            print("N is ={}".format(N))
        for idx2,xi in enumerate(Xnsp):
            
           neighbours = clf.kneighbors(xi.reshape(1,-1),
                                       n_neighbors=N, return_distance=False)[0]
           #print(neighbours)
           for idx3,ik in enumerate(neighbours):
              xki=Xnsp[ik,:] 
              xiP=np.vstack((xiP,np.array((list(xi+(np.random.random(1)*(xi-xki)))+
                                        [targetarry[targetNums.index(Ins)]]))))
              if verbose==True:
                print("{}.element of minority class {} has been created.".format(idx2+idx3,targetarry[targetNums.index(Ins)]))
    xiP2.append(xiP[1:-1,:])
    xiP=np.zeros((1,Xnp.shape[1]+1))
    return xiP2   
   
    
  def fit_resample(Xnp,target,N=-1,threshold=0.7,verbose=True):
       
       xiP=SMOTE.SMOTE(Xnp,target,N=N,threshold=threshold,verbose=verbose)
       for x in xiP:
           Xnp=np.vstack((Xnp,x[:,:-1]))
           target=np.array(list(target)+list(x[:,-1]))
       return Xnp,target