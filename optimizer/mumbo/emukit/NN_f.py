# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:16:20 2021

@author: fanny
"""


import numpy as np

# from  test_fun import sp, rastrigin,MFB1,E_MFB1,ackley,griewank,bas_mean,bas_min,forrester




class NN_sq: 
    def __init__(self,fun,type,b=0,a1=1,a0=0):

        self.fun=fun # 低可信度函数
        self.type=type
        self.b=b
        self.a0=a0
        self.a1=a1
        
    def __call__(self,x):
        a0=self.a0
        a1=self.a1
        
        if self.type=='sq':
            # y=np.sin(np.mean(x,1))*(self.fun(a1*x-self.b)**2)
            # y=1*(self.fun(a1*2*x-self.b)**2)
            
            y=np.sum(np.sin(2.*a1*x)*1,1)*(self.fun(a1*x-self.b)**2)
            # y=(np.sum(x,1) - np.sqrt(2)) * (self.fun(a1*x-self.b)) ** 2

            
            
            # y=self.fun(a1*x-self.b)**2+sp(x-np.sqrt(2)/2-a0)*(self.fun(a1*x-self.b)**2)
            # y=np.sin(self.fun(a1*x-self.b))
            
            
            
        elif self.type=='exp':
            # y=np.sqrt(np.sum(x**2,1))*1*np.exp(self.fun(self.a1*2*x-2-self.b))
            # y=np.sum(x,1)*(np.exp(self.fun(a1*2.*x-2-self.b)))-1
            y=1*np.exp(self.fun(self.a1*0.42*x-1-self.b))

        elif self.type=='sin':
            y=np.sin(self.fun(self.a1*1.02*x-1.2-self.b))-1#+ 1*self.fun(self.a1*0.42*abs(x)-0.3-self.b)


        elif self.type=='sqexp':
            y=0.2*np.sum(x,1)*(np.exp(self.fun(a1*2.*x-2-self.b)))-1+(np.sum(x,1) - np.sqrt(2)) * (self.fun(a1*x-self.b)) ** 2#+ 1*self.fun(self.a1*0.42*abs(x)-0.3-self.b)
            
                        
            
            
        return y



# def f_low(x):
#     return np.cos(15.*x)

# def f_high(x):
#     return x*np.exp(f_low(2.*x-2))-1




