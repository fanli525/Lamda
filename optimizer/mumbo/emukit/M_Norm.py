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


    def normalize(self, Y,Y0):
        self.mean=np.mean(Y0)
        self.std=np.std(Y0)
        
        if self.std !=0:
            return (Y-self.mean)/self.std
        else:
            return Y

    def inverse(self, X):

        if self.std !=0:
            return (X*self.std)+self.mean
        else:
            return X
        
        
    def inverse_variance(self, X):

        if self.std !=0:
            return (X*self.std**2)
        else:
            return X