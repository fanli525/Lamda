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


class MF_sLCB:
    def __init__(self, FM_name,model,next_task, w=0.5):
        self.model = model
        self.next_task = next_task
        self.FM_name = FM_name
        self.w = w
    def __call__(self, x):
        if self.FM_name== 'SGP':
            X = x.reshape(1,-1)
        else:
            X=np.zeros((1,len(x)+1))
            X[0,:-1]=x
            X[0,-1]=self.next_task
        py, var = self.model.predict(X)
        sigma = np.sqrt(abs(var))
        if np.min(sigma) > 0:
            lcb = py - self.w * sigma
        else:
            lcb = py
        self.lcb = lcb
        self.AC_value = lcb
        self.sigma = sigma
        return lcb[0]


class MEI_cost:
    def __init__(self, model, N_train,cost,d)-> None:
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
        self.model.task=2
        self.d=d

        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)

        n=x.shape[0]

        N_train=self.N_train
        
        X= convert_x_list_to_array([x, x])
        py, var = self.model.predict(X)

        
        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')
        self.model.task=2
        
        ei=np.zeros((n,self.model.task))
        
        for i in range(self.model.task):
             if i==0:
                y11=self.model.Y[0:N_train[0]]
             else:
                y11=self.model.Y[sum(N_train[0:i]):sum(N_train[0:i+1])]
             y11=self.model.Y[self.model.X[:,-1]==i]

             fmin=min(y11)
             Z = (fmin - py[:,i]) / sigma[:,i]
             ei[:,i] = (fmin - py[:,i]) * norm.cdf(Z) + sigma[:,i] * norm.pdf(Z)

        self.ei=ei        
        self.sigma=sigma

        ei_add_cost=ei/self.cost
        
        self.ei_add_cost=ei_add_cost
        
        self.AC_value=ei
        self.AC_cost_value=ei_add_cost
        
       
        if ei_add_cost[0][0]>ei_add_cost[0][1]:
            self.ind_task=0  
            return -ei_add_cost[:,0]
        else:
            self.ind_task=1  
            return -ei_add_cost[:,1]




class Max_EI:
    def __init__(self, model, N_train,cost,d):
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
        self.model.task=2
        self.d=d

    def __call__(self, x):
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)
        n=x.shape[0]

        N_train=self.N_train
        
 
            
        X= convert_x_list_to_array([x, x])
        
        py, var = self.model.predict(X)
        
        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')

        ei=np.zeros((n,self.model.task))
        
        for i in range(self.model.task):
             if i==0:
                y11=self.model.Y[0:N_train[0]]
             else:
                y11=self.model.Y[sum(N_train[0:i]):sum(N_train[0:i+1])]
             y11=self.model.Y[self.model.X[:,-1]==i]

             fmin=min(y11)
             Z = (fmin - py[:,i]) / sigma[:,i]
             ei[:,i] = (fmin - py[:,i]) * norm.cdf(Z) + sigma[:,i] * norm.pdf(Z)

        self.ei=ei
        self.AC_value=ei
        self.AC_cost_value=ei
        
        
        
        # ei_add_cost=self.cost*self.ei
        
        self.sigma=sigma
        
        if ei[0,0]>ei[0,1]:
            self.ind_task=0
        else:
            
            self.ind_task=1

        return -np.min(self.ei,1)



class Max_imEI:   # 利用单输出的预测不确定计算EI
    def __init__(self, model, N_train,cost,d,high_gp_model):
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
        self.model.task=2
        self.d=d
        self.high_gp_model=high_gp_model
    def __call__(self, x):
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)
        n=x.shape[0]

        N_train=self.N_train
 
        X= convert_x_list_to_array([x, x])
        
        py, var = self.model.predict(X)
        py=py.reshape(-1,self.model.task,order='F')

        hf_mean, hf_var  = self.high_gp_model.predict(x)
        hf_std = np.sqrt(hf_var) 
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')          

        sigma[:,self.model.task-1]=hf_std[:,0]



        ei=np.zeros((n,self.model.task))
        
        for i in range(self.model.task):
             if i==0:
                y11=self.model.Y[0:N_train[0]]
             else:
                y11=self.model.Y[sum(N_train[0:i]):sum(N_train[0:i+1])]
             y11=self.model.Y[self.model.X[:,-1]==i]

             fmin=min(y11)
             Z = (fmin - py[:,i]) / sigma[:,i]
             ei[:,i] = (fmin - py[:,i]) * norm.cdf(Z) + sigma[:,i] * norm.pdf(Z)

        self.ei=ei
        self.AC_value=ei
        self.AC_cost_value=ei
        
        
        
        # ei_add_cost=self.cost*self.ei
        
        self.sigma=sigma
        
        if ei[0,0]>ei[0,1]:
            self.ind_task=0
        else:
            
            self.ind_task=1

        return -np.min(self.ei,1)



class per_imEI:   # 利用单输出的预测不确定计算EI
    def __init__(self, model, N_train,cost,d,high_gp_model,ind_task,var_H):
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
        self.model.task=2
        self.d=d
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
 
        X= convert_x_list_to_array([x, x])
        try:
            py, var = self.model.predict(X)
        except:
            py, var = self.model.gpy_model.predict(X)
        # py, var = self.model.predict(X)
        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')          

        if self.var_H==1:
            hf_mean, hf_var  = self.high_gp_model.predict(x)
            hf_std = np.sqrt(hf_var) 
    
            sigma[:,self.model.task-1]=hf_std[:,0]



        ei=np.zeros((n,self.model.task))
        
        for i in range(self.model.task):
             if i==0:
                y11=self.model.Y[0:N_train[0]]
             else:
                y11=self.model.Y[sum(N_train[0:i]):sum(N_train[0:i+1])]
             y11=self.model.Y[self.model.X[:,-1]==i]
             fmin=min(y11)
             Z = (fmin - py[:,i]) / sigma[:,i]
             ei[:,i] = (fmin - py[:,i]) * norm.cdf(Z) + sigma[:,i] * norm.pdf(Z)

        self.ei=ei
        self.AC_value=ei
        self.AC_cost_value=ei
        
        
        
        # ei_add_cost=self.cost*self.ei
        
        self.sigma=sigma
        self.pre=py


        if self.ind_task==0:
            return -self.ei[:,0]
        else:
            return -self.ei[:,1]    

class per_imPI:   # 利用单输出的预测不确定计算EI
    def __init__(self, model, N_train,cost,d,high_gp_model,ind_task,var_H):
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
        self.model.task=2
        self.d=d
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
 
        X= convert_x_list_to_array([x, x])
        
        py, var = self.model.predict(X)
        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')          

        if self.var_H==1:
            hf_mean, hf_var  = self.high_gp_model.predict(x)
            hf_std = np.sqrt(hf_var) 
    
            sigma[:,self.model.task-1]=hf_std[:,0]



        poi=np.zeros((n,self.model.task))
        
        for i in range(self.model.task):
             if i==0:
                y11=self.model.Y[0:N_train[0]]
             else:
                y11=self.model.Y[sum(N_train[0:i]):sum(N_train[0:i+1])]
             y11=self.model.Y[self.model.X[:,-1]==i]

             fmin=min(y11)
             Z = (fmin - py[:,i]) / sigma[:,i]
             poi[:,i]  = norm.cdf(Z)

        self.poi=poi
        self.AC_value=poi
        self.AC_cost_value=poi
        
        
        
        # ei_add_cost=self.cost*self.ei
        
        self.sigma=sigma
        self.pre=py



        if self.ind_task==0:
            return -self.poi[:,0]
        else:
            return -self.poi[:,1]    



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
        self.model.task=2
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
 
        X= convert_x_list_to_array([x, x])
        try:
            py, var = self.model.predict(X)
        except:
            py, var = self.model.gpy_model.predict(X)

        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')          

        if self.var_H==1:
            hf_mean, hf_var  = self.high_gp_model.predict(x)
            hf_std = np.sqrt(hf_var) 
    
            sigma[:,self.model.task-1]=hf_std[:,0]


        if np.min(sigma)>0:        
            LCB_all=py - self.w * sigma
        else:
            LCB_all=py 

        self.AC_value=LCB_all
        self.AC_cost_value=LCB_all

 
        
        self.sigma=sigma
        self.pre=py



        if self.ind_task==0:
            return LCB_all[:,0]
        else:
            return LCB_all[:,1]    








 




class per_MB(Problem):   # 利用单输出的预测不确定计算EI

    def __init__(self, n_var,n_obj,xl,xu, model, d,high_gp_model,ind_task,var_H,N_train='',cost='',w=2,**kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, type_var=np.double, **kwargs)        


        self.model = model
        # self.N_train= N_train
        self.cost=cost
        self.model.task=2
        self.d=d
        self.w=w
        
        self.high_gp_model=high_gp_model
        self.ind_task =ind_task
        self.var_H=var_H
    def _evaluate(self, x,out, *args, **kwargs):
        
    # def __call__(self,x,out, *args, **kwargs):
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)
        n=x.shape[0]

        # N_train=self.N_train
 
        X= convert_x_list_to_array([x, x])
        try:
            py, var = self.model.predict(X)
        except:
            py, var = self.model.gpy_model.predict(X)
        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')          

        if self.var_H==1:
            hf_mean, hf_var  = self.high_gp_model.predict(x)
            hf_std = np.sqrt(hf_var) 
    
            sigma[:,self.model.task-1]=hf_std[:,0]


        if np.min(sigma)>0:        
            LCB_all=py - self.w * sigma
            LCB_all1=py - 4 * sigma

        else:
            LCB_all=py 
            LCB_all1=py 

        self.AC_value=LCB_all
        self.AC_cost_value=LCB_all


        ei=np.zeros((n,self.model.task))
        poi=np.zeros((n,self.model.task))
        
        for i in range(self.model.task):
             # if i==0:
             #    y11=self.model.Y[0:N_train[0]]
             # else:
             #    y11=self.model.Y[sum(N_train[0:i]):sum(N_train[0:i+1])]

             y11 = self.model.Y[self.model.X[:, -1] == i]

    #      fmin=min(y11)
        #      Z = (fmin - py[:,i]) / sigma[:,i]
        #      ei[:,i] = (fmin - py[:,i]) * norm.cdf(Z) + sigma[:,i] * norm.pdf(Z)
        #      poi[:,i]  = norm.cdf(Z)
        #
        #
        # self.ei=ei
        # self.AC_value=ei
        # self.AC_cost_value=ei

        self.sigma=sigma
        
        # if self.ind_task==0:
        #     out["F"] =np.column_stack([LCB_all[:,0],  LCB_all1[:,0]])
        # else:
        #     out["F"] =np.column_stack([LCB_all[:,1],  LCB_all1[:,0]])    

        if self.ind_task==0:
            out["F"] =np.column_stack([py[:,0],  -sigma[:,0]])
        else:
            out["F"] =np.column_stack([py[:,1],  -sigma[:,1]])   

    def _eval(self, x):
        
    # def __call__(self,x,out, *args, **kwargs):
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)
        n=x.shape[0]

        N_train=self.N_train
 
        X= convert_x_list_to_array([x, x])
        
        py, var = self.model.predict(X)
        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')          

        if self.var_H==1:
            hf_mean, hf_var  = self.high_gp_model.predict(x)
            hf_std = np.sqrt(hf_var) 
    
            sigma[:,self.model.task-1]=hf_std[:,0]

        if np.min(sigma)>0:        
            LCB_all=py - self.w * sigma
            LCB_all1=py - 4 * sigma

        else:
            LCB_all=py 
            LCB_all1=py 

        self.AC_value=LCB_all
        self.AC_cost_value=LCB_all


        ei=np.zeros((n,self.model.task))
        poi=np.zeros((n,self.model.task))
        
        for i in range(self.model.task):
             if i==0:
                y11=self.model.Y[0:N_train[0]]
             else:
                y11=self.model.Y[sum(N_train[0:i]):sum(N_train[0:i+1])]
             y11=self.model.Y[self.model.X[:,-1]==i]

             fmin=min(y11)
             Z = (fmin - py[:,i]) / sigma[:,i]
             ei[:,i] = (fmin - py[:,i]) * norm.cdf(Z) + sigma[:,i] * norm.pdf(Z)
             poi[:,i]  = norm.cdf(Z)

        self.ei=ei
        self.AC_value=ei
        self.AC_cost_value=ei

        self.sigma=sigma
        


        # if self.ind_task==0:
        #     return np.column_stack([LCB_all[:,0],  LCB_all1[:,0]])
        # else:
        #     return np.column_stack([LCB_all[:,1],  LCB_all1[:,0]])

        if self.ind_task==0:
            return np.column_stack([py[:,0],  -sigma[:,0]])
        else:
            return np.column_stack([py[:,1],  -sigma[:,1]])

# import EMOC

# class MyProblem(EMOC.Problem):
#
#     def __init__(self, dec_num, obj_num,xl,xu,model, N_train,cost,d,w,high_gp_model,ind_task,var_H,**kwargs):
#         super(MyProblem, self).__init__(dec_num,obj_num)
#         self.lower_bound = xl
#         self.upper_bound = xu
#         self.count = 0
#         self.model=model
#
#         self.N_train= N_train
#         self.cost=cost
#         self.model.task=2
#         self.d=d
#         self.w=2
#
#         self.high_gp_model=high_gp_model
#         self.ind_task =ind_task
#         self.var_H=var_H
#
#
#     def CalObj(self, ind):
#         self.count = self.count + 1
#         x = ind.dec
#         x = np.atleast_2d(x)
#
#         x=x.reshape(-1,self.d)
#         n=x.shape[0]
#
#         N_train=self.N_train
#
#         X= convert_x_list_to_array([x, x])
#
#         py, var = self.model.predict(X)
#         py=py.reshape(-1,self.model.task,order='F')
#         sigma= np.sqrt(abs(var))
#         sigma=sigma.reshape(-1,self.model.task,order='F')
#
#         if self.var_H==1:
#             hf_mean, hf_var  = self.high_gp_model.predict(x)
#             hf_std = np.sqrt(hf_var)
#
#             sigma[:,self.model.task-1]=hf_std[:,0]
#
#
#         # if np.min(sigma)>0:
#         #     LCB_all=py - self.w * sigma
#         #     LCB_all1=py - 4 * sigma
#
#         # else:
#         #     LCB_all=py
#         #     LCB_all1=py
#
#         # self.AC_value=LCB_all
#         # self.AC_cost_value=LCB_all
#
#
#         # ei=np.zeros((n,self.model.task))
#         # poi=np.zeros((n,self.model.task))
#
#         # for i in range(self.model.task):
#         #       if i==0:
#         #         y11=self.model.Y[0:N_train[0]]
#         #       else:
#         #         y11=self.model.Y[sum(N_train[0:i]):sum(N_train[0:i+1])]
#
#         #       fmin=min(y11)
#         #       Z = (fmin - py[:,i]) / sigma[:,i]
#         #       ei[:,i] = (fmin - py[:,i]) * norm.cdf(Z) + sigma[:,i] * norm.pdf(Z)
#         #       poi[:,i]  = norm.cdf(Z)
#
#
#         # self.ei=ei
#         # self.AC_value=ei
#         # self.AC_cost_value=ei
#
#         self.sigma=sigma
#
#         temp_obj=np.zeros((x.shape[0],self.obj_num))
#         if self.ind_task==0:
#             temp_obj[:,0]=py[:,0]
#             temp_obj[:,1]=-sigma[:,0]
#             ind.obj = list(temp_obj[0])
#         else:
#             temp_obj[:,0]=py[:,1]
#             temp_obj[:,1]=-sigma[:,1]
#             ind.obj = list(temp_obj[0])
#
#     def _eval(self, x):
#
#     # def __call__(self,x,out, *args, **kwargs):
#         """
#         Computes the Expected Improvement.
#
#         :param x: points where the acquisition is evaluated.
#         """
#         x=x.reshape(-1,self.d)
#         n=x.shape[0]
#
#         N_train=self.N_train
#
#         X= convert_x_list_to_array([x, x])
#
#         py, var = self.model.predict(X)
#         py=py.reshape(-1,self.model.task,order='F')
#         sigma= np.sqrt(abs(var))
#         sigma=sigma.reshape(-1,self.model.task,order='F')
#
#         if self.var_H==1:
#             hf_mean, hf_var  = self.high_gp_model.predict(x)
#             hf_std = np.sqrt(hf_var)
#
#             sigma[:,self.model.task-1]=hf_std[:,0]
#
#         # if np.min(sigma)>0:
#         #     LCB_all=py - self.w * sigma
#         #     LCB_all1=py - 4 * sigma
#
#         # else:
#         #     LCB_all=py
#         #     LCB_all1=py
#
#         # self.AC_value=LCB_all
#         # self.AC_cost_value=LCB_all
#
#
#         # ei=np.zeros((n,self.model.task))
#         # poi=np.zeros((n,self.model.task))
#
#         # for i in range(self.model.task):
#         #       if i==0:
#         #         y11=self.model.Y[0:N_train[0]]
#         #       else:
#         #         y11=self.model.Y[sum(N_train[0:i]):sum(N_train[0:i+1])]
#
#         #       fmin=min(y11)
#         #       Z = (fmin - py[:,i]) / sigma[:,i]
#         #       ei[:,i] = (fmin - py[:,i]) * norm.cdf(Z) + sigma[:,i] * norm.pdf(Z)
#         #       poi[:,i]  = norm.cdf(Z)
#
#         # self.ei=ei
#         # self.AC_value=ei
#         # self.AC_cost_value=ei
#
#         self.sigma=sigma
#
#
#
#         # if self.ind_task==0:
#         #     return np.column_stack([LCB_all[:,0],  LCB_all1[:,0]])
#         # else:
#         #     return np.column_stack([LCB_all[:,1],  LCB_all1[:,0]])
#
#         if self.ind_task==0:
#             return np.column_stack([py[:,0],  -sigma[:,0]])
#         else:
#             return np.column_stack([py[:,1],  -sigma[:,0]])
#
#
#
# class MyProblem_cost(EMOC.Problem):
#
#     def __init__(self, dec_num, obj_num,xl,xu,model, N_train,cost,d,w,high_gp_model,var_H,**kwargs):
#         super(MyProblem_cost, self).__init__(dec_num,obj_num)
#         self.lower_bound = xl
#         self.upper_bound = xu
#         self.count = 0
#         self.model=model
#
#         self.N_train= N_train
#         self.cost=cost
#         self.model.task=2
#         self.d=d
#         self.w=2
#
#         self.high_gp_model=high_gp_model
#         self.var_H=var_H
#         self.cost=cost
#
#     def CalObj(self, ind):
#         self.count = self.count + 1
#         cost=self.cost
#
#
#         x = ind.dec
#         x = np.atleast_2d(x)
#         ind1=(x[:,-1]>=0.5)
#         x[:, -1] = int(np.zeros(x.shape[0]))
#         x[ind1, -1]=int(np.ones(len(ind1)))
#         ind.dec= list(x[0])
#
#
#
#
#         x=x.reshape(-1,self.d)
#         n=x.shape[0]
#
#         N_train=self.N_train
#
#         X=  x
#
#         py, var = self.model.predict(X)
#         sigma= np.sqrt(abs(var))
#
#         self.sigma=sigma
#
#         temp_obj=np.zeros((x.shape[0],self.obj_num))
#         if x[:,-1]==0:
#             if py>0:
#                 temp_obj[:,0]=py*cost[0]
#             else:
#                 temp_obj[:,0]=py/cost[0]
#
#             temp_obj[:,1]=-sigma/cost[0]
#             ind.obj = list(temp_obj[0])
#         else:
#             if py>0:
#                 temp_obj[:,0]=py*cost[1]
#             else:
#                 temp_obj[:,0]=py/cost[1]
#             temp_obj[:,1]=-sigma/cost[1]
#             ind.obj = list(temp_obj[0])
#
#     def _eval(self, x):
#
#     # def __call__(self,x,out, *args, **kwargs):
#         """
#         Computes the Expected Improvement.
#
#         :param x: points where the acquisition is evaluated.
#         """
#         cost=self.cost
#
#         x = np.atleast_2d(x)
#         ind1 = (x[:, -1] >= 0.5)
#         x[:, -1] = np.zeros(x.shape[0])
#         x[ind1, -1] = np.ones(len(ind1))
#
#         x=x.reshape(-1,self.d)
#         n=x.shape[0]
#
#         N_train=self.N_train
#
#         X=  x
#
#         py, var = self.model.predict(X)
#         sigma= np.sqrt(abs(var))
#
#         self.sigma=sigma
#
#         temp_obj=np.zeros((x.shape[0],self.obj_num))
#         if x[-1]==0:
#             if py>0:
#                 temp_obj[:,0]=py*cost[0]
#             else:
#                 temp_obj[:,0]=py/cost[0]
#
#             temp_obj[:,1]=-sigma[:,0]/cost[0]
#             return np.column_stack([temp_obj[:,0], temp_obj[:,1]])
#         else:
#             if py>0:
#                 temp_obj[:,0]=py*cost[1]
#             else:
#                 temp_obj[:,0]=py/cost[1]
#             temp_obj[:,1]=-sigma[:,1]/cost[1]
#
#             return np.column_stack([temp_obj[:,0], temp_obj[:,1]])
#
#
 


# class SF_MyProblem(EMOC.Problem):
#
#     def __init__(self, dec_num, obj_num,xl,xu,model, d,w,**kwargs):
#         super(SF_MyProblem, self).__init__(dec_num,obj_num)
#         self.lower_bound = xl
#         self.upper_bound = xu
#         self.count = 0
#         self.model=model
#
#         self.d=d
#         self.w=2
#
#
#
#
#     def CalObj(self, ind):
#         self.count = self.count + 1
#         x = ind.dec
#         x = np.atleast_2d(x)
#
#         x=x.reshape(-1,self.d)
#         n=x.shape[0]
#
#
#
#         py, var = self.model.predict(x)
#         sigma= np.sqrt(abs(var))
#
#
#         self.sigma=sigma
#
#         temp_obj=np.zeros((x.shape[0],self.obj_num))
#         temp_obj=py
#         temp_obj=-sigma
#         ind.obj = list(temp_obj[0])
#
#
#     def _eval(self, x):
#
#     # def __call__(self,x,out, *args, **kwargs):
#         """
#         Computes the Expected Improvement.
#
#         :param x: points where the acquisition is evaluated.
#         """
#         x=x.reshape(-1,self.d)
#         n=x.shape[0]
#         N_train=self.N_train
#         py, var = self.model.predict(x)
#         sigma= np.sqrt(abs(var))
#         self.sigma=sigma
#         return np.column_stack([py,  -sigma])













class per_MIN:   # 利用单输出的预测不确定计算EI
    def __init__(self, model, N_train,cost,d):
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
        self.model.task=2
        self.d=d

        
    def __call__(self, x):
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)
        n=x.shape[0]

        N_train=self.N_train
 
        X= convert_x_list_to_array([x, x])
        
        py, var = self.model.predict(X)
        py=py.reshape(-1,self.model.task,order='F')

        return py[:,1]    




















class Max_Var:
    def __init__(self, model, N_train,cost,d):
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
        self.model.task=2
        self.d=d

    def __call__(self, x):
        x=x.reshape(-1,self.d)

        n=x.shape[0]

        N_train=self.N_train
        X= convert_x_list_to_array([x, x])

        py, var = self.model.predict(X)
        
        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')
        self.model.task=2
        
        ei=np.zeros((n,self.model.task))
        
        for i in range(self.model.task):
             if i==0:
                y11=self.model.Y[0:N_train[0]]
             else:
                y11=self.model.Y[sum(N_train[0:i]):sum(N_train[0:i+1])]
             y11=self.model.Y[self.model.X[:,-1]==i]

             fmin=min(y11)
             Z = (fmin - py[:,i]) / sigma[:,i]
             ei[:,i] = (fmin - py[:,i]) * norm.cdf(Z) + sigma[:,i] * norm.pdf(Z)

        self.ei=ei
        # ei_add_cost=self.cost*self.ei
        
        self.sigma=sigma
        
        
        self.AC_value=sigma
        self.AC_cost_value=sigma
        
        if sigma[0,0]>sigma[0,1]:
            self.ind_task=0
        else:
            
            self.ind_task=1        
        
        return -np.max(self.sigma,1)




class MLCB_cost:
    def __init__(self, model, N_train,cost,d,w = 2.0)-> None:
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
        self.w=w
        self.model.task=2
        self.d=d
       
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)

        n=x.shape[0]

        N_train=self.N_train
        X= convert_x_list_to_array([x, x])
        py, var = self.model.predict(X)

        
        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')
        self.model.task=2
        
        self.sigma=sigma
        
        if np.min(sigma)>0:        
            LCB_all=py - self.w * sigma
        else:
            LCB_all=py 
  

        if np.max(LCB_all)<0:
            LCB_all_1=LCB_all/self.cost
        else:
            LCB_all_1=LCB_all*self.cost

        self.LCB_all=LCB_all
        self.LCB_cost=LCB_all_1
        

        
        if n>1:
            max_=np.max(LCB_all,1)
            max_ind=(max_<0)
            
            self.LCB_cost=LCB_all*self.cost
            self.LCB_cost[max_ind,:]=LCB_all[max_ind,:]/self.cost

            
        self.AC_value=LCB_all
        self.AC_cost_value=self.LCB_cost
        # if sigma[0,0]<0.1 and sigma[0,1]<0.1:
        #     LCB_all_1= -sigma/self.cost          
        
        if  LCB_all.shape[0]==1:
            # if sigma[0,0]<0.1 and sigma[0,1]>0.1:
            #     self.ind_task=1
            # elif sigma[0,0]>0.1 and sigma[0,1]<0.1:
                
            #     self.ind_task=0

            # else:
            a = LCB_all_1
            a=a.reshape(2, 1, order='F')
            sorted_id = sorted(range(len(a)), key=lambda k:a[k], reverse=False)
            self.ind_task= sorted_id[0]
            
        return np.min(LCB_all_1,1)
        


class Min_LCB:
    def __init__(self, model, N_train, cost,d,w=2):
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
        self.model.task=2

        self.w=w
        self.d=d

        
        
    def __call__(self, x):
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)

        n=x.shape[0]

        N_train=self.N_train
        X= convert_x_list_to_array([x, x])

        py, var = self.model.predict(X)
        
        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')
        self.model.task=2

        if np.min(sigma)>0:        
            LCB_all=py - self.w * sigma
        else:
            LCB_all=py 


        self.lcb=LCB_all
        
        self.AC_value=LCB_all
        self.AC_cost_value=LCB_all
        
        
        self.sigma=sigma
        
        if LCB_all[0,0]<LCB_all[0,1]:
            self.ind_task=0
        else:
            
            self.ind_task=1

        
        return np.min(self.lcb,1)



class Min_LCB_SGP:
    def __init__(self, model, N_train, cost,d,w=2):
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
        self.task=2

        self.w=w
        self.d=d

        
        
    def __call__(self, x):
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)

        n=x.shape[0]
        py=np.zeros((n,2))
        var=np.zeros((n,2))
        
        py[:,0],var[:,0]=self.model[0].predict(x)
        py[:,1],var[:,1]=self.model[1].predict(x) 
        



        # N_train=self.N_train
        # X= convert_x_list_to_array([x, x])

        # py, var = self.model.predict(X)
        
        # py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.task,order='F')

        if np.min(sigma)>0:        
            LCB_all=py - self.w * sigma
        else:
            LCB_all=py 


        self.lcb=LCB_all
        
        self.AC_value=LCB_all
        self.AC_cost_value=LCB_all
        
        
        self.sigma=sigma
        
        if LCB_all[0,0]<LCB_all[0,1]:
            self.ind_task=0
        else:
            
            self.ind_task=1

        
        return np.min(self.lcb,1)










class Min_LCB_bandit:
    def __init__(self, model, N_train,cost,d,w = 2.0)-> None:
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
        self.w=w
        self.model.task=2
        self.d=d
       
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)

        n=x.shape[0]

        N_train=self.N_train
        X= convert_x_list_to_array([x, x])
        py, var = self.model.predict(X)

        
        py=py.reshape(-1,self.model.task,order='F')
        sigma= np.sqrt(abs(var))
        sigma=sigma.reshape(-1,self.model.task,order='F')
        self.model.task=2
        
        self.sigma=sigma
        
        if np.min(sigma)>0:        
            LCB_all=py - self.w * sigma
        else:
            LCB_all=py 
  



        self.LCB_all=LCB_all
        
        self.AC_value=LCB_all
        self.AC_cost_value=LCB_all
        
        
        

        if sigma[0,0]>1.0e-1:
            self.ind_task=0
        else:
            
            self.ind_task=1      
        

            
        return np.max(LCB_all,1)
        




class SGP_LCB:
    def __init__(self, model, N_train, cost,d,w=2):
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
        self.w=w
        self.model.task=2
        self.d=d
       
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x=x.reshape(-1,self.d)

        n=x.shape[0]

        N_train=self.N_train
        
        
        py, var = self.model.predict(x)

        

        sigma= np.sqrt(abs(var))

        self.model.task=2
        
        self.sigma=sigma
        
        if np.min(sigma)>0:        
            LCB_all=py - self.w * sigma
        else:
            LCB_all=py 
  



        self.LCB_all=LCB_all
        
        self.AC_value=LCB_all
        self.AC_cost_value=LCB_all
        

  
        return LCB_all[:,0]


class SGP_EI:
    def __init__(self, model, N_train, cost, d, w=2):
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
        self.N_train = N_train
        self.cost = cost
        self.w = w
        self.model.task = 2
        self.d = d

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        x = x.reshape(-1, self.d)

        n = x.shape[0]

        N_train = self.N_train


        py, var = self.model.predict(x)



        sigma= np.sqrt(abs(var))


        y11=self.model.Y

        fmin=min(y11)
        Z = (fmin - py) / sigma
        ei = (fmin - py) * norm.cdf(Z) + sigma * norm.pdf(Z)

        self.ei=ei
        self.sigma=sigma



        return -ei[:,0]












