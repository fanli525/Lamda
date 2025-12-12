# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 10:16:28 2021

@author: fanny
"""

import numpy as np
import scipy as sp

def stable_Cholesky(K):

    try:
        L = np.linalg.cholesky(K);
    except:
        pass
    
      
    
    diagPower = np.min( [np.ceil(np.log10(abs(np.min(np.diag(K)))))-1, -11]);
    if ~(abs(diagPower) < np.inf):
        diagPower = -10
    
    success = 0;
    K = K + (10**diagPower) * np.eye(K.shape[0],K.shape[1]);
    while success==0:
        try:
            L = np.linalg.cholesky(K);
            success = 1
        except:
    
            if diagPower > 1e-3:
                print('CHOL failed with diagPower = %d\n', diagPower);
                K, 
            diagPower = diagPower + 1; 
            K = K + (10**diagPower) * np.eye(K.shape[0],K.shape[1]);
    
    
    return L

