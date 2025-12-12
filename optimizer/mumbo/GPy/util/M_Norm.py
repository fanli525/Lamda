# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 12:11:04 2021

对y进行归一化
@author: fl347
"""
import numpy as np
class Norm:
    def __init__(self):
        self.mean = None
        self.std = None


    def normalize(self, Y):
        self.mean=np.mean(Y)
        self.std=np.std(Y)
        

        return (Y-self.mean)/(self.std+1.0e-8)


    def inverse(self, X):

        if self.std >0:
            return (X*self.std+1.0e-8)+self.mean
        else:
            return X
        
        
        
    def inverse_variance(self, X):

        if self.std !=0:
            return (X*self.std**2)
        else:
            return X           