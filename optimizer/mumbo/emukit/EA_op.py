# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:38:37 2021

@author: fanny
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:58:30 2021

@author: fl347
"""
import numpy as np
# from numpy.random import multivariate_normal #For later example

# import pandas as pd
import time
import math

from numpy import where, sum, any, mean, array, clip, ones, abs
from numpy.random import uniform, choice, normal, randint, random, rand
from copy import deepcopy
from scipy.stats import cauchy
import matplotlib.pyplot as plt

# import GPyOpt

from  errors import root_mean_squared_error
from scipy.spatial.distance import cdist

from init_latin_hypercube_sampling  import  init_latin_hypercube_sampling

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import datetime

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_problem

from pymoo.core.problem import Problem
from pymoo.core.evaluator import set_cv
from pymoo.util.termination.no_termination import NoTermination


import GPy
import mock
import pytest

from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement,MaxValueEntropySearch,EntropySearch
from emukit.core.interfaces import IModel
from emukit.core.parameter_space import ParameterSpace
from emukit.core.continuous_parameter import ContinuousParameter

from emukit.core.loop import UserFunctionWrapper, FixedIterationsStoppingCondition

from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper











class Sgp_lcbga1: 
    def __init__(self,dimension,problem_bounds,fo,fo_name,phi,N_train,maxN,AC_name,r):
        self.dimension =dimension
        self.problem_bounds =problem_bounds
        self.fo =fo
        
        self.fo_name =fo_name
        self.AC_name=AC_name

        self.phi = phi
        self.N_train=N_train
        self.maxN = maxN
        self.r=r
    def __call__(self):
        #初始化问题信息
        import numpy as np

        np.random.seed(self.r)
        
        dimension=self.dimension
        problem_bounds=self.problem_bounds
        fo=self.fo 
        AC_name=self.AC_name

        fo_name=self.fo_name 

        phi= self.phi
        maxN= min(self.maxN*dimension,1000)
        N_train=11*dimension-1
        d=dimension
        # 初始化模型的参数
        # Generate Normalized Dataset of (x,y) values
        N_test=1000
        N_train0=N_train
        
        if dimension==1:
                        
            import numpy as np
                        

            xtest =np.linspace(problem_bounds[0], problem_bounds[1], N_test)
            xtest = np.asarray([[xi] for xi in xtest])
        else: 
                        
            import numpy as np

            xtest= np.random.uniform(problem_bounds[0], problem_bounds[1], size=(N_test, dimension))
            

        ytest=fo(xtest)

        
        it=0
        # 初始化PSO的参数
        if d<5:
            npop=40
        else:
            npop=80  
        
        beta = 1
        pc = 1
        gamma = 0.1
        mu = 0.01
        sigma = 0.1

        
        Tbd=np.ones(d)*problem_bounds[0]
        Tbu=np.ones(d)*problem_bounds[1]
        
        Tbd_pop=np.repeat(Tbd[np.newaxis, :], npop, axis=0)
        Tbu_pop=np.repeat(Tbu[np.newaxis, :], npop, axis=0)
        tvmax=0.5*(Tbu_pop-Tbd_pop)
        # Xt=self.Xt
        Xt=init_latin_hypercube_sampling(Tbd, Tbu, N_train0,np.random.RandomState(self.r))

        Yt=fo(Xt)
        Yt=Yt[:,np.newaxis]

        
        
        Samp=Xt
        YS=Yt


        
        pop=init_latin_hypercube_sampling(Tbd, Tbu, npop,np.random.RandomState(self.r))

        

        N_IC=np.array([1])# 采样点的数目
        it=0


        Ymin=np.min(Yt[:,0])

        
        Ac=np.zeros(0)


        D_best_op=np.zeros(0)#  当前最优值到真值最优值的欧式距离
        D_new_op=np.zeros(0)#  当前采样点到真值最优值的欧式距离
        D_mean_op=np.zeros(0)#  当前种群均值到真值最优值的欧式距离
        D_diver=np.zeros(0)#  当前种群的分散度


        space = ParameterSpace([ContinuousParameter('x'+str(i), Tbd[i], Tbu[i]) for i in range(d)])

        # RBF
        kernel = GPy.kern.Matern52(input_dim=d)

        gpy_model = GPy.models.GPRegression(Samp, YS,kernel,normalizer=True)
        # gpy_model['.*Mat52.var'].constrain_bounded(lower=1.0e-4, upper=1.0e4)# constrain_positive()m['.*StdPeriodic.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.optimizer='trust_region'
        gpy_model['.*noise*'].constrain_fixed(1.0e-6)

        model = GPyModelWrapper(gpy_model,n_restarts = 1)
        model.optimize()

        mean, var= model.model.predict(pop)
                 

        fpop=mean
        fgbest=fpop[0]
        gbest=pop[0,:]  
        Tc = 1
        NE=np.zeros(0)
        Ac=np.zeros(0)
        Ac_px=np.zeros(0)
        dalta=1e-6
        
        nc = int(np.round(pc*npop/2)*2)
        Ac_ini=np.zeros(0)        

        if self.fo_name==1:
            x_ropt=np.ones(d)*0
            
        Tbd_in=x_ropt-(Tbu-Tbd)/4
        Tbu_in=x_ropt+(Tbu-Tbd)/4
        
        
        Xt_ini=init_latin_hypercube_sampling(Tbd_in, Tbu_in, 80,np.random.RandomState(self.r))
        Yt_ini=fo(Xt_ini)            




        # set the meta-data of the problem (necessary to initialize the algorithm)
        problem = Problem(n_var=d, n_obj=1, n_constr=0, xl=Tbd, xu=Tbu)
        
        # create the algorithm object
        algorithm = GA(pop_size=npop)
        
        # let the algorithm object never terminate and let the loop control it
        termination = NoTermination()
        
        # create an algorithm object that never terminates
        algorithm.setup(problem, termination=termination)


        EV_inf=[]


        pop1=pop
            
        T1=time.time()

        while N_train<maxN and np.max(np.std(pop1,0))>1.0e-6 and Tc==1:
            
            pop = algorithm.ask()
            pop1 = pop.get("X")


            if AC_name=='ES':
                acquisition =EntropySearch(model,space) 
            elif AC_name=='MES':
                acquisition =MaxValueEntropySearch(model,space) 
            else:
                pass
            
            y_LCB_s =acquisition.evaluate(pop1)#y_LCB_s = mean - LCB_w * sigma
            
           # 由大到小排序
            sorted_id = sorted(range(len(y_LCB_s)), key=lambda k: y_LCB_s[k], reverse=True)
            

            for ic in range(N_IC[0]):
            
                index =sorted_id[ic]
                x_new=pop1[index,:]
                x_new=np.asarray([x_new])
                xa= np.asarray(Xt)
                dist =cdist(x_new,xa,metric='euclidean')
                if np.min(dist)> dalta:
                    index =np.random.randint(0,len(sorted_id))
                    x_new = pop1[index, :]
                    x_new = np.asarray([x_new])

                    y_new=fo(x_new)


                    N_train=N_train+1
                    Xt=np.vstack((Xt,x_new))
                    Yt=np.append(Yt,y_new)
                    Ymin=np.append(Ymin,np.min(Yt))
    
                    Samp=np.vstack((Samp,x_new))
                    YS=np.append(YS,y_new)
                    YS=YS[:,np.newaxis]

                    d1=np.sqrt(np.sum((gbest-x_ropt)**2))
                    D_best_op=np.append(D_best_op, d1)#  当前最优值到真值最优值的欧式距离
            
                    d1=np.sqrt(np.sum((x_new-x_ropt)**2))
                    D_new_op=np.append(D_new_op, d1)#  当前最优值到真值最优值的欧式距离
            
                    d1=np.sqrt(np.sum((np.mean(pop1,0)-x_ropt)**2))
                    D_mean_op=np.append(D_mean_op, d1)#  当前最优值到真值最优值的欧式距离
            
            
            
                    d1=(pop1-np.mean(pop1,0))**2
                    d1=np.sqrt(np.sum(d1,1))
                    L=np.sqrt(np.sum((Tbu-Tbd)**2));
                    d1=np.sum(d1)/(L*npop)
            
                    D_diver=np.append(D_diver, d1)#  当前最优值到真值最优值的欧式距离
 



            # RBF
            kernel = GPy.kern.Matern52(input_dim=d)
    
            gpy_model = GPy.models.GPRegression(Samp, YS,kernel,normalizer=True)
            # gpy_model['.*Mat52.var'].constrain_bounded(lower=1.0e-4, upper=1.0e4)# constrain_positive()m['.*StdPeriodic.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.optimizer='trust_region'
            gpy_model['.*noise*'].constrain_fixed(1.0e-6)
    
            model = GPyModelWrapper(gpy_model,n_restarts = 1)
            model.optimize()
            
            
            x0=pop1
            
            mean, var= model.model.predict(x0)


            if math.isnan(max(mean)) == False:
            # 更新gbest和pbest
                fpop=mean

                # objectives
                pop.set("F", np.column_stack([fpop]))
            
                # this line is necessary to set the CV and feasbility status - even for unconstrained
                set_cv(pop)
            
                # returned the evaluated individuals which have been evaluated or even modified
                algorithm.tell(infills=pop)
        


                sorted_id = sorted(range(len(fpop)), key=lambda k:fpop[k], reverse=False)


                Ib=np.where(fpop==np.min(fpop))[0][0]
                if fgbest>fpop[Ib]:
                    fgbest=fpop[Ib]
                    gbest=pop1[Ib,:]          
        
                fpop_real=fo(pop1)
                sorted_id = sorted(range(len(fpop_real)), key=lambda k: fpop_real[k], reverse=False)
                ind_r=np.arange(0, npop, 1)
        
                a=(np.array(sorted_id)==ind_r)
                px=np.sum(a==1)/len(a)
                Ac_px=np.append(Ac_px, px)#  当前最优值到真值最优值的欧式距离        
        
                m_ini, var_ini=model.model.predict(Xt_ini)
        
                sorted_pin = sorted(range(len(m_ini)), key=lambda k: m_ini[k], reverse=False)
                
                sorted_rin = sorted(range(len(Yt_ini)), key=lambda k: Yt_ini[k], reverse=False)
                
        
                a=(np.array(sorted_pin)==np.array(sorted_rin))
                px=np.sum(a==1)/len(a)
                Ac_ini=np.append(Ac_ini, px)#  当前最优值到真值最优值的欧式距离        
                                

            #   测试算法的精度

                mean, var= model.model.predict(xtest)

                #  accurate test
                #Error1=error(ytest,mean, disp=1)
                b=np.array(root_mean_squared_error(ytest,mean[:,0]))
                Ac=np.append(Ac,b)
                print("N_train:", N_train)

                it=it+1
                NE= np.append(NE, N_train)
                n1 = NE.shape[0]
                if n1>30:

                    if np.std(NE[n1-30:,],0) ==0:

                        Tc=0


        #         if dimension==1:
        #             plt.figure()
        #             plt.plot(xtest,ytest, 'k--',label='real fun' )
        #             plt.plot(xtest,mean, 'r--',label='pre fun' )
        #             plt.plot(pop1,fpop, 'bo',label='POP points' )
            
        #             plt.plot(Samp,YS, 'ko',label='Train points' )
        #             plt.plot(Samp[-1],YS[-1], 'rs',label='Train points' )
            
        #             plt.show()


        #         if dimension==2:
        #             theta0 = np.linspace(problem_bounds[0], problem_bounds[1], 100)
        #             theta1 = np.linspace(problem_bounds[0], problem_bounds[1], 100)
        #             Theta0, Theta1 = np.meshgrid(theta0, theta1)
                 
                
        #             LML=np.zeros((Theta0.shape[0],Theta0.shape[1]))
        #             for i in range(Theta0.shape[0]):
        #                 for j in range(Theta0.shape[1]):  
        #                     LML[i,j]=model.model.predict(np.array([Theta0[i, j], Theta1[i, j]]).reshape(-1,2))[0]
                
                
                
                
                
                
        #             if it%1 ==0:
        #                 plt.figure()
        #                 CS=plt.contour(Theta1,Theta0 , LML)#,8,alpha=0.75,cmap=plt.cm.hot
                        
                        
        #                 plt.clabel(CS,inline=1,fontsize=10)
        #                 # ax.scatter(x[:,0],x[:,1], y, marker='o',s=6.0,color=(1,0,0.),label='Train points')  
        #                 plt.scatter(Samp[:,0],Samp[:,1], marker='o',s=30.0,color=(0,0,0.),label='Train points')  
        #                 plt.scatter(Samp[-1,0],Samp[-1,1], marker='s',s=50.0,color=(1,0,0.),label='Train points') 
        #                 plt.scatter(pop1[:,0],pop1[:,1], marker='o',s=20.0,color=(0,0,1.),label='Train points')  
                        
        #                 plt.scatter(0,0, marker='*',s=190.0,color=(1,0,0.),label='Train points') 
        #                 plt.title('Nt = '+str(N_train))
                
                        
        #                 cb=plt.colorbar()
        #                 plt.show()
             
        # plt.figure()
        # plt.subplot(411)
        # plt.plot(D_best_op, 'k--',label='D_best_op' )
        # plt.legend()      
        
        # plt.subplot(412)
        # plt.plot(D_new_op, 'k--',label='D_new_op' )
        # plt.legend()      
        
        # plt.subplot(413)
        # plt.plot(D_mean_op, 'k--',label='D_mean_op' )
        # plt.legend()      
        
        # plt.subplot(414)
        # plt.plot(D_diver, 'k--',label='D_diver' )
        
            
        # plt.legend()      
        # plt.tight_layout()
        
        # plt.figure()
        # plt.subplot(211)
        
        # plt.plot(Ac_px, 'k--',label='Ac_px' )
        # plt.subplot(212)
        
        # plt.plot(Ac_ini, 'k--',label='Ac_ini' )
        
        # plt.legend()    
        
        
        # plt.show()



        EV_inf.append({'D_best_op': D_best_op,'D_new_op': D_new_op,'D_mean_op': D_mean_op,
                       'D_diver': D_diver,'Ac_px': Ac_px,'Ac_ini': Ac_ini})
 
        Ib=np.where(YS==np.min(Yt))[0][0]
        x_opt=Samp[Ib,:]
        y_opt=np.min(Yt)
        T2=time.time()
        return x_opt,y_opt,Samp,YS[:,0],Ac,Ymin,T2 - T1,EV_inf









class Sgp_lcbDE: 
    def __init__(self,dimension,problem_bounds,fo,fo_name,phi,N_train,maxN,AC_name,r):
        self.dimension =dimension
        self.problem_bounds =problem_bounds
        self.fo =fo
        
        self.fo_name =fo_name

        self.phi = phi
        self.N_train=N_train
        self.maxN = maxN
        self.AC_name =AC_name
        self.r=r
    def __call__(self):
        #初始化问题信息
        np.random.seed(self.r)
        
        dimension=self.dimension
        problem_bounds=self.problem_bounds
        fo=self.fo 
        
        fo_name=self.fo_name 
        AC_name=self.AC_name

        phi= self.phi
        maxN= min(self.maxN*dimension,1000)
        N_train=11*dimension-1
        d=dimension
        # 初始化模型的参数
        # Generate Normalized Dataset of (x,y) values
        N_test=1000
        N_train0=N_train
        
        if dimension==1:
            xtest =np.linspace(problem_bounds[0], problem_bounds[1], N_test)
            xtest = np.asarray([[xi] for xi in xtest])
        else: 
            xtest= np.random.uniform(problem_bounds[0], problem_bounds[1], size=(N_test, dimension))
            

        ytest=fo(xtest)

        
        it=0
        # 初始化PSO的参数
        if d<5:
            n=40
        else:
            n=80  
        
        self.weighting_factor = 0.8
        self.crossover_rate = 0.9

        
        Tbd=np.ones(d)*problem_bounds[0]
        Tbu=np.ones(d)*problem_bounds[1]
        
        Tbd_pop=np.repeat(Tbd[np.newaxis, :], n, axis=0)
        Tbu_pop=np.repeat(Tbu[np.newaxis, :], n, axis=0)
        tvmax=0.5*(Tbu_pop-Tbd_pop)
        # Xt=self.Xt
        Xt=init_latin_hypercube_sampling(Tbd, Tbu, N_train0,np.random.RandomState(self.r))

        Yt=fo(Xt)
        Yt=Yt[:,np.newaxis]

        
        
        Samp=Xt
        YS=Yt


        
        pop=init_latin_hypercube_sampling(Tbd, Tbu, n,np.random.RandomState(self.r))

        

        N_IC=np.array([1])# 采样点的数目
        it=0
        Ymin=np.min(Yt)
        space = ParameterSpace([ContinuousParameter('x'+str(i), Tbd[i], Tbu[i]) for i in range(d)])

        # RBF
        kernel = GPy.kern.Matern52(input_dim=d)

        gpy_model = GPy.models.GPRegression(Samp, YS,kernel,normalizer=True)
        # gpy_model['.*Mat52.var'].constrain_bounded(lower=1.0e-4, upper=1.0e4)# constrain_positive()m['.*StdPeriodic.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.optimizer='trust_region'
        gpy_model['.*noise*'].constrain_fixed(1.0e-6)

        model = GPyModelWrapper(gpy_model,n_restarts = 1)
        model.optimize()
        
        
        
        mean, var= model.model.predict(pop) 
                    

        fpop=mean 
        Ib=np.where(fpop==np.min(fpop))[0][0]
        fgbest=fpop[Ib]
        gbest=pop[Ib,:]  
         
 
        Tc = 1
        NE=np.zeros(0)
        
        D_best_op=np.zeros(0)#  当前最优值到真值最优值的欧式距离
        D_new_op=np.zeros(0)#  当前采样点到真值最优值的欧式距离
        D_mean_op=np.zeros(0)#  当前种群均值到真值最优值的欧式距离
        D_diver=np.zeros(0)#  当前种群的分散度
        Ac=np.zeros(0)
        Ac_px=np.zeros(0)
        Ac_ini=np.zeros(0)                
                        
                
        dalta=1e-8
        
        if self.fo_name==1:
            x_ropt=np.ones(d)*0
        T1=time.time()
        EV_inf=[]

        Tbd_in = x_ropt - (Tbu - Tbd) / 4
        Tbu_in = x_ropt + (Tbu - Tbd) / 4

        Xt_ini = init_latin_hypercube_sampling(Tbd_in, Tbu_in, 80,np.random.RandomState(self.r))
        Yt_ini = fo(Xt_ini)

        while N_train<maxN and Tc==1:
            


            pop_child=np.zeros((n,d))
            for i in range(0, n):
                idx_list = choice(list(set(range(0, n)) - {i}), 2, replace=False)

                pos_new = pop[i,:]+ self.weighting_factor * (gbest- pop[i,:]) + \
                          self.weighting_factor * (pop[idx_list[0],:] - pop[idx_list[1],:])     
                # pos_new = self._mutation__(pop[i,:], pos_new)
                
                pos_new = where(uniform(0, 1, d) < self.crossover_rate, pos_new,pop[i,:] )
                pos_new =np.clip(pos_new, Tbd, Tbu)
                
                
                pop_child[i,:] = pos_new
            pop1=np.vstack((pop,pop_child))


            if AC_name=='ES':
                acquisition =EntropySearch(model,space) 
            elif AC_name=='MES':
                acquisition =MaxValueEntropySearch(model,space) 
            else:
                pass


            y_LCB_s =acquisition.evaluate(pop1)#y_LCB_s = mean - LCB_w * sigma
            
           # 由大到小排序
            sorted_id = sorted(range(len(y_LCB_s)), key=lambda k: y_LCB_s[k], reverse=True)
            

            for ic in range(N_IC[0]):
            
                index =sorted_id[ic]
                x_new=pop1[index,:]
                x_new=np.asarray([x_new])
                xa= np.asarray(Xt)
                dist =cdist(x_new,xa,metric='euclidean')
                if np.min(dist)<dalta:
                    index =sorted_id[1]
                    x_new=pop1[index,:]
                    x_new=np.asarray([x_new])
                    xa= np.asarray(Xt)
                    dist =cdist(x_new,xa,metric='euclidean')
                if np.min(dist)>=dalta:

                    y_new=fo(x_new)
                    
                    
                    N_train=N_train+1
                    Xt=np.vstack((Xt,x_new))
                    Yt=np.append(Yt,y_new)
                    Ymin = np.append(Ymin, np.min(Yt))

                    Samp=np.vstack((Samp,x_new))
                    YS=np.append(YS,y_new)
                    YS=YS[:,np.newaxis]
                    
                    d1=np.sqrt(np.sum((gbest-x_ropt)**2))
                    D_best_op=np.append(D_best_op, d1)#  当前最优值到真值最优值的欧式距离
        
                    d1=np.sqrt(np.sum((x_new-x_ropt)**2))
                    D_new_op=np.append(D_new_op, d1)#  当前最优值到真值最优值的欧式距离
        
                    d1=np.sqrt(np.sum((np.mean(pop,0)-x_ropt)**2))
                    D_mean_op=np.append(D_mean_op, d1)#  当前最优值到真值最优值的欧式距离
        
        
        
                    d1=(pop-np.mean(pop,0))**2
                    d1=np.sqrt(np.sum(d1,1))
                    L=np.sqrt(np.sum((Tbu-Tbd)**2));
                    d1=np.sum(d1)/(L*n)
        
                    D_diver=np.append(D_diver, d1)#  当前最优值到真值最优值的欧式距离


                    

            # RBF
            kernel = GPy.kern.Matern52(input_dim=d)
    
            gpy_model = GPy.models.GPRegression(Samp, YS,kernel,normalizer=True)
            # gpy_model['.*Mat52.var'].constrain_bounded(lower=1.0e-4, upper=1.0e4)# constrain_positive()m['.*StdPeriodic.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.optimizer='trust_region'
            gpy_model['.*noise*'].constrain_fixed(1.0e-6)
    
            model = GPyModelWrapper(gpy_model,n_restarts = 1)
            model.optimize()
            x0=pop1

            mean, var= model.model.predict(x0)



            fpop=mean[0:n,0]
            fpop_child=mean[n:,0]            
            

            
            
            if math.isnan(max(mean)) == False:
                fx0=fo(x0)
                fpop_real=fx0[0:n]
                fpop_child_real=fx0[n:]
                px_p=fpop<fpop_child
                px_r=fpop_real<fpop_child_real
                a=(px_p==px_r)
                px=np.sum(a==1)/len(a)
                Ac_px=np.append(Ac_px, px)#  当前最优值到真值最优值的欧式距离        
                
                m_ini, var_ini= model.model.predict(Xt_ini)
 
        
                sorted_pin = sorted(range(len(m_ini)), key=lambda k: m_ini[k], reverse=False)
                
                sorted_rin = sorted(range(len(Yt_ini)), key=lambda k: Yt_ini[k], reverse=False)
                
        
                a=(np.array(sorted_pin)==np.array(sorted_rin))
                px=np.sum(a==1)/len(a)
                Ac_ini=np.append(Ac_ini, px)#  当前最优值到真值最优值的欧式距离        
                        

                pop = [pop[i,:] if fpop[i] < fpop_child[i] else pop_child[i,:] for i in range(n)]
                fpop = [fpop[i] if fpop[i] < fpop_child[i] else fpop_child[i] for i in range(n)]
                
                pop=np.array(pop)
                fpop=np.array(fpop)
                Ib=np.where(fpop==np.min(fpop))[0][0]
                if fgbest>fpop[Ib]:
                    fgbest=fpop[Ib]
                    gbest=pop[Ib,:]            
                                 
                            
                
                mean, var= model.model.predict(xtest)
                                  

                #  accurate test
                #Error1=error(ytest,mean, disp=1)
                b=np.array(root_mean_squared_error(ytest,mean[:,0]))
                Ac=np.append(Ac,b)
                print("N_train:", N_train)

                it=it+1
                NE= np.append(NE, N_train)
                n1 = NE.shape[0]
                if n1>30:

                    if np.std(NE[n1-30:,],0) ==0:

                        Tc=0



        #         if dimension==1:
        #             plt.figure()
        #             plt.plot(xtest,ytest, 'k--',label='real fun' )
        #             plt.plot(xtest,mean, 'r--',label='pre fun' )
        #             plt.plot(pop,fpop, 'bo',label='Train points' )
            
        #             plt.plot(Samp,YS, 'ko',label='Train points' )
        #             plt.plot(Samp[-1],YS[-1], 'rs',label='Train points' )
            
        #             plt.show()


        #         if dimension==2:
        #             theta0 = np.linspace(problem_bounds[0], problem_bounds[1], 100)
        #             theta1 = np.linspace(problem_bounds[0], problem_bounds[1], 100)
        #             Theta0, Theta1 = np.meshgrid(theta0, theta1)
                 
                
        #             LML=np.zeros((Theta0.shape[0],Theta0.shape[1]))
        #             for i in range(Theta0.shape[0]):
        #                 for j in range(Theta0.shape[1]):  
        #                     LML[i,j]=model.model.predict(np.array([Theta0[i, j], Theta1[i, j]]).reshape(-1,2))[0]

                
                
                
                
                
        #             if it%1 ==0:
        #                 plt.figure()
        #                 CS=plt.contour(Theta1,Theta0 , LML)#,8,alpha=0.75,cmap=plt.cm.hot
                        
                        
        #                 plt.clabel(CS,inline=1,fontsize=10)
        #                 # ax.scatter(x[:,0],x[:,1], y, marker='o',s=6.0,color=(1,0,0.),label='Train points')  
        #                 plt.scatter(Samp[:,0],Samp[:,1], marker='o',s=30.0,color=(0,0,0.),label='Train points')  
        #                 plt.scatter(Samp[-1,0],Samp[-1,1], marker='s',s=50.0,color=(1,0,0.),label='Train points') 
        #                 plt.scatter(pop[:,0],pop[:,1], marker='o',s=20.0,color=(0,0,1.),label='Train points')  
                        
        #                 plt.scatter(0,0, marker='*',s=190.0,color=(1,0,0.),label='Train points') 
        #                 plt.title('Nt = '+str(N_train))
                
                        
        #                 cb=plt.colorbar()
        #                 plt.show()
             
        # plt.figure()
        # plt.subplot(411)
        # plt.plot(D_best_op, 'k--',label='D_best_op' )
        # plt.legend()      
        
        # plt.subplot(412)
        # plt.plot(D_new_op, 'k--',label='D_new_op' )
        # plt.legend()      
        
        # plt.subplot(413)
        # plt.plot(D_mean_op, 'k--',label='D_mean_op' )
        # plt.legend()      
        
        # plt.subplot(414)
        # plt.plot(D_diver, 'k--',label='D_diver' )
        
            
        # plt.legend()      
        # plt.tight_layout()
        
        # plt.figure()
        # plt.subplot(211)
        
        # plt.plot(Ac_px, 'k--',label='Ac_px' )
        # plt.subplot(212)
        
        # plt.plot(Ac_ini, 'k--',label='Ac_ini' )
        
        # plt.legend()    
        
        
        # plt.show()




        EV_inf.append({'D_best_op': D_best_op,'D_new_op': D_new_op,'D_mean_op': D_mean_op,
                       'D_diver': D_diver,'Ac_px': Ac_px,'Ac_ini': Ac_ini})
             
        Ib=np.where(YS==np.min(Yt))[0][0]
        x_opt=Samp[Ib,:]
        y_opt=np.min(Yt)
        T2=time.time()
        return x_opt,y_opt,Samp,YS[:,0],Ac,Ymin,T2 - T1,EV_inf















def crossover(p1, p2, gamma=0.1):
    c1 = p1
    c2 = p1
    alpha = np.random.uniform(-gamma, 1+gamma, *c1.shape)
    c1 = alpha*p1 + (1-alpha)*p2
    c2 = alpha*p2 + (1-alpha)*p1
    return c1, c2

def mutate(x, mu, sigma):
    X=x
    n=X.shape[0]
    Y=np.zeros((n,X.shape[1]))
    for i in range(n):
        x=X[i,:]
        y = x
        flag = np.random.rand(*x.shape) <= mu
        ind = np.argwhere(flag)
        y[ind] += sigma*np.random.randn(*ind.shape)
        Y[i,:]=y
    return Y

    # y = x
    # flag = np.random.rand(*x.shape) <= mu
    # ind = np.argwhere(flag)
    # y[ind] += sigma*np.random.randn(*ind.shape)

    # return y
def apply_bound(x, varmin, varmax):
    x= np.maximum(x, varmin)
    x = np.minimum(x, varmax)

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]



def roulette_select(population, fitnesses, num):
    """ Roulette selection, implemented according to:
        <http://stackoverflow.com/questions/177271/roulette
        -selection-in-genetic-algorithms/177278#177278>
    """
    # total_fitness = float(sum(fitnesses))
    # rel_fitness = [f/total_fitness for f in fitnesses]
    # # Generate probability intervals for each individual
    # probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
    # # Draw new population
    # new_population = []
    # for n in xrange(num):
    #     r = rand()
    #     for (i, individual) in enumerate(population):
    #         if r <= probs[i]:
    #             new_population.append(individual)
    #             break
    # return new_population

    total_fitness = float(sum(fitnesses))
    rel_fitness = [f/total_fitness for f in fitnesses]
    # Generate probability intervals for each individual
    probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
    # Draw new population
    new_population = []
    for n in range(num):
        r = rand()
        for (i, individual) in enumerate(population):
            if r <= probs[i]:
                new_population.append(individual)
                break
    return new_population