# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:25:23 2022
ranking loss
@author: fl347
"""


import numpy as np

def rank_loss( x,y):
    n=len(x)
    loss=0
    par_all=0
    par_ri=0
    ind=np.arange(0,n,1)
    for i in range(n):
        for j in ind[i+1:]:
            par_all=par_all+1
            if ( (x[i]<=x[j]) and (y[i]<=y[j])  or  (x[i]>=x[j]) and (y[i]>=y[j])):
                par_ri=par_ri+1

            else:
                loss=loss+1
                
    return loss,par_ri,par_all
    
