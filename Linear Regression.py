#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

def getMSE(X,Y,W):
    '''
    This fuction is used to get the mean square error value between 
    the model prediction(Fx) and the actual value(Y) for a instance(Xi,Yi) 
    in the dataset
    
    returns the MSE : Mean square error value
    '''
    # Calculating MSE
    
    Fx = (X).dot(W.T)
    MSE = (1/(X.shape[0]))*(sum((Y-Fx)**2))
  
    return MSE[0]

def fitData(X,Y,W,sigma2):
    '''
    Runs gradient descent for the given iterations and optimises the weights to fit
    the data.
    
    Returns W : optimised parameters, 
    Ein : Error for in sample data, 
    Eout: Error for out sample data
    '''
    # Performing gradient descent
    
    N,d = X.shape
    for i in range(500):
        Fx = (X).dot(W.T)
        W = W + ((0.003)*(2/N)*((Y - Fx).T).dot(X))
    
    # calculating Ein and Eout
    
    Ein = getMSE(X,Y,W)
    X_out,Y_out = getData(1000,d-1,sigma2)
    Eout = getMSE(X_out,Y_out,W)
    
    return W,Ein,Eout    


def getData(N,d,sigma2):
    '''
    This fuction generates data based on the  given parameters
    N = no of rows, sigma2 = variance , d = Degree of polynomial
    
    returns X : data,
    Y : labels of actual values
    '''
    
    # Calculating X upto the given degree
    
    x= np.random.rand(N,1)
    X = np.zeros((N,1))
    for i in range(d+1):
        X= np.concatenate([X,x**i],axis = 1)
   
    X=np.delete(X,0,axis=1)
   
    # Calculating Z and Y 
    
    if d > 0:
        Z = np.random.normal(loc = 0.0,scale =sigma2)
        Y=np.cos(2*np.pi*X[:,1:2]) +Z
    else:
        Z = np.random.normal(loc = 0.0,scale =sigma2)
        Y=np.cos(2*np.pi*X[:,:1]) +Z
    
    return X,Y

def experiment(N,d,sigma2):
    '''
    This takes in aruguments N: No of training examples ,
    d: Degree of Polynomial ,
    sigma2: Variance
    
    returns E_bias: ,
    Ein_avg : Average error for in sample data, 
    Eout_avg: Average error for out sample data
    '''
    # Intialising required variables
    
    W = np.zeros((1,d+1))
    Ein =list()
    Eout =list()

    # Performing experiment for M trials
    
    for i in range(50):
        X,Y = getData(N,d,sigma2)
        w = np.random.rand(1,X.shape[1])
        w, ein, eout = fitData(X,Y,w,sigma2)
        W = np.concatenate([W,w])
        Ein.append(ein)
        Eout.append(eout)

    # Calculating Ein_avg,Eout_avg and averahe of thr M polynomials
    
    W = np.delete(W,0,0)
    Ein_avg = np.mean(Ein)
    Eout_avg = np.mean(Eout)
    W_avg = np.mean(W,axis =0)

    # Calculating the E_bias
    
    X,Y = getData(2000,d,sigma2)
    E_bias = getMSE(X,Y,W_avg)

    return E_bias,Ein_avg,Eout_avg


# Declaring the required variables

N,d,sigma2 = [2, 5, 10, 20, 50, 100, 200],[*range(0,21)],[0.01**2,0.1**2,1]
data = np.zeros((1,6))

# Running the experiment fuction for all combination of N,d,sigma2

for i in N:
    for j in d:
        for k in sigma2:
            E_bias,Ein,Eout = experiment(i,j,k)
            d1 = np.array([i,j,k,E_bias,Ein,Eout]).reshape(1,6)
            data = np.concatenate([data,d1])

# Transforming the collected data into Dataframe for plotting

collected_data = np.delete(data,0,0)
final_data = pd.DataFrame(collected_data,
                          columns=['No of examples','Degree of polynomial','Variance','E_bias','Ein', 'Eout'])

# Saving collected data

final_data.to_csv("Final Data.csv")