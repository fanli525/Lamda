# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:38:08 2022

@author: fl347
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:01:35 2021

@author: fl347
"""




from typing import Tuple, Union

import scipy.stats
import numpy as np
from scipy.stats import norm
# import itertools

from emukit.core.interfaces import IModel, IModelWithNoise, IDifferentiable, IJointlyDifferentiable
from emukit.core.acquisition import Acquisition

from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

import GPy


from scipy.stats import norm
from pymoo.core.problem import Problem
from pymoo.util.normalization import normalize





class per_imLCB:   # 利用单输出的预测不确定计算EI
    def __init__(self, model, N_train,cost,d,w,high_gp_model,ind_task,var_H):
        """
        For a given input, this acquisition computes the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """

        self.model = model
        self.N_train= N_train
        self.cost=cost
        self.model.task=int(max(self.model.X[:,-1])+1)
        self.d=d
        self.w=w
        
        self.high_gp_model=high_gp_model
        self.ind_task =ind_task
        self.var_H=var_H
        
    def __call__(self, x):
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)
        n=x.shape[0]

        N_train=self.N_train
 
        X=  convert_x_list_to_array([x for i in range(self.model.task)])
        
        py, var = self.model.predict(X)
        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')          

        if np.min(sigma)>0:        
            LCB_all=py - self.w * sigma
        else:
            LCB_all=py 

        self.AC_value=LCB_all
        self.AC_cost_value=LCB_all
        self.sigma=sigma
        
        return LCB_all[:,self.ind_task]


    def evaluate(self, X):
        py, var = self.model.predict(X)
        py = py.reshape(-1, self.model.task, order='F')
        sigma = np.sqrt(abs(var))
        sigma = sigma.reshape(-1, self.model.task, order='F')



        if np.min(sigma) > 0:
            LCB_all = py - self.w * sigma
        else:
            LCB_all = py

        self.AC_value = LCB_all
        self.AC_cost_value = LCB_all

        self.sigma = sigma

        return LCB_all











class Cost_MB(Problem):   # 利用单输出的预测不确定计算EI

    def __init__(self, n_var,n_obj,xl,xu, model,w,Cost_fun,**kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, type_var=np.double, **kwargs)        


        self.model = model
        self.d=n_var
        self.w=2
        self.Cost_fun=Cost_fun
    def _evaluate(self, x,out, *args, **kwargs):
        
    # def __call__(self,x,out, *args, **kwargs):
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """

 
        
        py, var = self.model.predict(x)
        sigma= np.sqrt(abs(var))

        if np.min(sigma)>0:        
            LCB_all=py - self.w * sigma
        else:
            LCB_all=py 

        self.AC_value=LCB_all
        self.AC_cost_value=LCB_all
        self.sigma=sigma  

        cost=self.Cost_fun.evaluate(x)
        out["F"] =np.column_stack([LCB_all,  cost])   

    def _eval(self, x):
        
    # def __call__(self,x,out, *args, **kwargs):
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """

        
        py, var = self.model.predict(x)
        sigma= np.sqrt(abs(var))

        if np.min(sigma)>0:        
            LCB_all=py - self.w * sigma
        else:
            LCB_all=py 

        self.AC_value=LCB_all
        self.AC_cost_value=LCB_all
        self.sigma=sigma

        cost=self.Cost_fun.evaluate(x)


        return np.column_stack([LCB_all,  cost])




























