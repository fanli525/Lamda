# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 23:25:36 2022

@author: fanny
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 20:01:40 2021

@author: fl347

"""




import numpy as np
from init_latin_hypercube_sampling  import  init_latin_hypercube_sampling
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
np.random.seed(0)


 


class PSO: 
    def __init__(self,dimension,problem_bounds,fo,maxiter):

        self.dimension=dimension
        self.problem_bounds =problem_bounds

        self.fo=fo
        self.maxiter=maxiter
 
        
    def __call__(self):
        
        fun=self.fo
        
        d=self.dimension
        problem_bounds=self.problem_bounds
 
 
        Tbd=problem_bounds[0]
        Tbu=problem_bounds[1]

        n=20
        if d>2:
            n=40
        w=0.729;   
        c1=1.491;  
        c2=1.491;
        wmax=0.729;
        wmin=.2;

        
        Tbd_pop=np.repeat(Tbd[np.newaxis, :], n, axis=0)
        Tbu_pop=np.repeat(Tbu[np.newaxis, :], n, axis=0)
        tvmax=0.5*(Tbu_pop-Tbd_pop)
        pop=init_latin_hypercube_sampling(Tbd, Tbu, n)
        
        u=init_latin_hypercube_sampling(Tbd, Tbu, n)
        u=0.25*u; v=u;
        
        lbest=pop
        fpop=fun(pop)
        fbest=fpop
        Ib = np.where(fbest == np.min(fbest))[0][0]
        gbest=lbest[Ib,:]
        fgbest=fbest[Ib]
        it=0
        while it<self.maxiter:


            w=wmax-(wmax-wmin)*it/self.maxiter
            v=w*v+c1*np.random.rand(n,d)*(lbest-pop)+c2*np.random.rand(n,d)*(gbest-pop)
            
            
            v[v>tvmax]=tvmax[v>tvmax]
            v[v<(0-tvmax)]=0-tvmax[v<(0-tvmax)]
            tPOP=pop+v
            
            tPOP[tPOP>Tbu_pop]=Tbu_pop[tPOP>Tbu_pop]-(tPOP[tPOP>Tbu_pop]-Tbu_pop[tPOP>Tbu_pop])
            tPOP[tPOP<Tbd_pop]=Tbd_pop[tPOP<Tbd_pop]+(Tbd_pop[tPOP<Tbd_pop]-tPOP[tPOP<Tbd_pop])
            
            pop=tPOP       



 
            fpop=fun(pop)

            ind=(fpop<=fbest)
            fbest[ind]=fpop[ind]
            lbest[ind,:]=pop[ind,:]
            Ib = np.where(fbest == np.min(fbest))[0][0]


            if fbest[Ib]<=fgbest:
                gbest=lbest[Ib,:]
                fgbest=fbest[Ib]

  

            it=it+1
 

 
        return gbest,fgbest
    
