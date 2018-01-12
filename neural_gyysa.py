#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 02:33:17 2018

@author: sdrttnclskn
"""
import numpy as np

# sigmoid function

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


# input dataset
    
X = np.array([  [0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1] ])
    

# output dataset
    
y = np.array([[0,0,1,1]]).T


np.random.seed(1)

# agirliklar

syn0 = 2*np.random.random((3,1)) - 1 # -1,1

for iter in range(100000):
   
    # ileri besleme
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))
    
    # hata
    l1_error = y - l1
    
    l1_delta = l1_error * sigmoid(l1,True)
    
    # agirlik güncelleme
    syn0 += np.dot(l0.T,l1_delta)
    
    
print ("Güncel Agirliklar")
print (syn0)
print ("Cikis")
print (l1)
print ("Hata")
print ((y - l1)/ 100)
