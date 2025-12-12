# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 02:38:41 2021

@author: fanny
"""
import numpy as np
# from numpy.random import multivariate_normal #For later example

# import pandas as pd
import time
import math
import matplotlib.pyplot as plt
from numpy import where, sum, any, mean, array, clip, ones, abs
from numpy.random import uniform, choice, normal, randint, random, rand
from copy import deepcopy
from scipy.stats import cauchy


# import GPyOpt

from  errors import root_mean_squared_error
from scipy.spatial.distance import cdist

from init_latin_hypercube_sampling  import  init_latin_hypercube_sampling


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













import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 01:32:05 2021

@author: fanny
"""


import math
import numpy as np

from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32

class CMA:
    """CMA-ES stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::

           import numpy as np
           from cmaes import CMA

           def quadratic(x1, x2):
               return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

           optimizer = CMA(mean=np.zeros(2), sigma=1.3)

           for generation in range(50):
               solutions = []
               for _ in range(optimizer.population_size):
                   # Ask a parameter
                   x = optimizer.ask()
                   value = quadratic(x[0], x[1])
                   solutions.append((x, value))
                   print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")

               # Tell evaluation values.
               optimizer.tell(solutions)

    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter (optional).

        n_max_resampling:
            A maximum number of resampling parameters (default: 100).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.

        seed:
            A seed number (optional).

        population_size:
            A population size (optional).

        cov:
            A covariance matrix (optional).
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cov: Optional[np.ndarray] = None,
    ):
        assert sigma > 0, "sigma must be non-zero positive value"

        assert np.all(
            np.abs(mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

        n_dim = len(mean)
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))  # (eq. 48)
        assert population_size > 0, "popsize must be non-zero positive value."

        mu = population_size // 2

        # (eq.49)
        weights_prime = np.array(
            [
                math.log((population_size + 1) / 2) - math.log(i + 1)
                for i in range(population_size)
            ]
        )
        mu_eff = (np.sum(weights_prime[:mu]) ** 2) / np.sum(weights_prime[:mu] ** 2)
        mu_eff_minus = (np.sum(weights_prime[mu:]) ** 2) / np.sum(
            weights_prime[mu:] ** 2
        )

        # learning rate for the rank-one update
        alpha_cov = 2
        c1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
        # learning rate for the rank-μ update
        cmu = min(
            1 - c1 - 1e-8,  # 1e-8 is for large popsize.
            alpha_cov
            * (mu_eff - 2 + 1 / mu_eff)
            / ((n_dim + 2) ** 2 + alpha_cov * mu_eff / 2),
        )
        assert c1 <= 1 - cmu, "invalid learning rate for the rank-one update"
        assert cmu <= 1 - c1, "invalid learning rate for the rank-μ update"

        min_alpha = min(
            1 + c1 / cmu,  # eq.50
            1 + (2 * mu_eff_minus) / (mu_eff + 2),  # eq.51
            (1 - c1 - cmu) / (n_dim * cmu),  # eq.52
        )

        # (eq.53)
        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
        weights = np.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            min_alpha / negative_sum * weights_prime,
        )
        cm = 1  # (eq. 54)

        # learning rate for the cumulation for the step-size control (eq.55)
        c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + c_sigma
        assert (
            c_sigma < 1
        ), "invalid learning rate for cumulation for the step-size control"

        # learning rate for cumulation for the rank-one update (eq.56)
        cc = (4 + mu_eff / n_dim) / (n_dim + 4 + 2 * mu_eff / n_dim)
        assert cc <= 1, "invalid learning rate for cumulation for the rank-one update"

        self._n_dim = n_dim
        self._popsize = population_size
        self._mu = mu
        self._mu_eff = mu_eff

        self._cc = cc
        self._c1 = c1
        self._cmu = cmu
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma
        self._cm = cm

        # E||N(0, I)|| (p.28)
        self._chi_n = math.sqrt(self._n_dim) * (
            1.0 - (1.0 / (4.0 * self._n_dim)) + 1.0 / (21.0 * (self._n_dim ** 2))
        )

        self._weights = weights

        # evolution path
        self._p_sigma = np.zeros(n_dim)
        self._pc = np.zeros(n_dim)

        self._mean = mean

        if cov is None:
            self._C = np.eye(n_dim)
        else:
            assert cov.shape == (n_dim, n_dim), "Invalid shape of covariance matrix"
            self._C = cov

        self._sigma = sigma
        self._D: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

        # bounds contains low and high of each parameter.
        assert bounds is None or _is_valid_bounds(bounds, mean), "invalid bounds"
        self._bounds = bounds
        self._n_max_resampling = n_max_resampling

        self._g = 0
        self._rng = np.random.RandomState(seed)

        # Termination criteria
        self._tolx = 1e-12 * sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * n_dim / population_size)
        self._funhist_values = np.empty(self._funhist_term * 2)

    def __getstate__(self) -> Dict[str, Any]:
        attrs = {}
        for name in self.__dict__:
            # Remove _rng in pickle serialized object.
            if name == "_rng":
                continue
            if name == "_C":
                sym1d = _compress_symmetric(self._C)
                attrs["_c_1d"] = sym1d
                continue
            attrs[name] = getattr(self, name)
        return attrs

    def __setstate__(self, state: Dict[str, Any]) -> None:
        state["_C"] = _decompress_symmetric(state["_c_1d"])
        del state["_c_1d"]
        self.__dict__.update(state)
        # Set _rng for unpickled object.
        setattr(self, "_rng", np.random.RandomState())

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._n_dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    def set_bounds(self, bounds: Optional[np.ndarray]) -> None:
        """Update boundary constraints"""
        assert bounds is None or _is_valid_bounds(bounds, self._mean), "invalid bounds"
        self._bounds = bounds

    def ask(self) -> np.ndarray:
        """Sample a parameter"""
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_feasible(x):
                return x
        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        return x

    def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._B is not None and self._D is not None:
            return self._B, self._D

        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D ** 2)), B.T)

        self._B, self._D = B, D
        return B, D

    def _sample_solution(self) -> np.ndarray:
        B, D = self._eigen_decomposition()
        z = self._rng.randn(self._n_dim)  # ~ N(0, I)
        y = cast(np.ndarray, B.dot(np.diag(D))).dot(z)  # ~ N(0, C)
        x = self._mean + self._sigma * y  # ~ N(m, σ^2 C)
        return x

    def _is_feasible(self, param: np.ndarray) -> bool:
        if self._bounds is None:
            return True
        return cast(
            bool,
            np.all(param >= self._bounds[:, 0]) and np.all(param <= self._bounds[:, 1]),
        )  # Cast bool_ to bool.

    def _repair_infeasible_params(self, param: np.ndarray) -> np.ndarray:
        if self._bounds is None:
            return param

        # clip with lower and upper bound.
        param = np.where(param < self._bounds[:, 0], self._bounds[:, 0], param)
        param = np.where(param > self._bounds[:, 1], self._bounds[:, 1], param)
        return param

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""

        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

        self._g += 1
        solutions.sort(key=lambda s: s[1])

        # Stores 'best' and 'worst' values of the
        # last 'self._funhist_term' generations.
        funhist_idx = 2 * (self.generation % self._funhist_term)
        self._funhist_values[funhist_idx] = solutions[0][1]
        self._funhist_values[funhist_idx + 1] = solutions[-1][1]

        # Sample new population of search_points, for k=1, ..., popsize
        B, D = self._eigen_decomposition()
        self._B, self._D = None, None

        x_k = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)
        y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)

        # Selection and recombination
        y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)  # eq.41
        self._mean += self._cm * self._sigma * y_w

        # Step-size control
        C_2 = cast(
            np.ndarray, cast(np.ndarray, B.dot(np.diag(1 / D))).dot(B.T)
        )  # C^(-1/2) = B D^(-1) B^T
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * C_2.dot(y_w)

        norm_p_sigma = np.linalg.norm(self._p_sigma)
        self._sigma *= np.exp(
            (self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1)
        )
        self._sigma = min(self._sigma, _SIGMA_MAX)

        # Covariance matrix adaption
        h_sigma_cond_left = norm_p_sigma / math.sqrt(
            1 - (1 - self._c_sigma) ** (2 * (self._g + 1))
        )
        h_sigma_cond_right = (1.4 + 2 / (self._n_dim + 1)) * self._chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)

        # (eq.45)
        self._pc = (1 - self._cc) * self._pc + h_sigma * math.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * y_w

        # (eq.46)
        w_io = self._weights * np.where(
            self._weights >= 0,
            1,
            self._n_dim / (np.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + _EPS),
        )

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)  # (p.28)
        assert delta_h_sigma <= 1

        # (eq.47)
        rank_one = np.outer(self._pc, self._pc)
        rank_mu = np.sum(
            np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0
        )
        self._C = (
            (
                1
                + self._c1 * delta_h_sigma
                - self._c1
                - self._cmu * np.sum(self._weights)
            )
            * self._C
            + self._c1 * rank_one
            + self._cmu * rank_mu
        )

    def should_stop(self) -> bool:
        B, D = self._eigen_decomposition()
        dC = np.diag(self._C)

        # Stop if the range of function values of the recent generation is below tolfun.
        if (
            self.generation > self._funhist_term
            and np.max(self._funhist_values) - np.min(self._funhist_values)
            < self._tolfun
        ):
            return True

        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        if np.all(self._sigma * dC < self._tolx) and np.all(
            self._sigma * self._pc < self._tolx
        ):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(D) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(dC))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self.generation % self.dim
        if np.all(self._mean == self._mean + (0.1 * self._sigma * D[i] * B[:, i])):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(D) / np.min(D)
        if condition_cov > self._tolconditioncov:
            return True

        return False


def _is_valid_bounds(bounds: Optional[np.ndarray], mean: np.ndarray) -> bool:
    if bounds is None:
        return True
    if (mean.size, 2) != bounds.shape:
        return False
    if not np.all(bounds[:, 0] <= mean):
        return False
    if not np.all(mean <= bounds[:, 1]):
        return False
    return True


def _compress_symmetric(sym2d: np.ndarray) -> np.ndarray:
    assert len(sym2d.shape) == 2 and sym2d.shape[0] == sym2d.shape[1]
    n = sym2d.shape[0]
    dim = (n * (n + 1)) // 2
    sym1d = np.zeros(dim)
    start = 0
    for i in range(n):
        sym1d[start : start + n - i] = sym2d[i][i:]  # noqa: E203
        start += n - i
    return sym1d


def _decompress_symmetric(sym1d: np.ndarray) -> np.ndarray:
    n = int(np.sqrt(sym1d.size * 2))
    assert (n * (n + 1)) // 2 == sym1d.size
    R, C = np.triu_indices(n)
    out = np.zeros((n, n), dtype=sym1d.dtype)
    out[R, C] = sym1d
    out[C, R] = sym1d
    return out








class Sgp_CMAES: 
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
        AC_name =self.AC_name

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
         
        bestsol_cost=fpop[Ib]
        Tc = 1
        NE=np.zeros(0)

        bounds=np.zeros((d,2))
        bounds[:,0] = Tbd
        bounds[:,1] = Tbu
        D_best_op=np.zeros(0)#  当前最优值到真值最优值的欧式距离
        D_new_op=np.zeros(0)#  当前采样点到真值最优值的欧式距离
        D_mean_op=np.zeros(0)#  当前种群均值到真值最优值的欧式距离
        D_diver=np.zeros(0)#  当前种群的分散度        
        Ac=np.zeros(0)
        Ac_px=np.zeros(0)
        Ac_ini=np.zeros(0)        
        optimizer = CMA(mean=np.zeros(d), sigma=1.3,bounds=bounds,population_size=n)

        dalta=1e-6
        

            
        if self.fo_name==1:
            x_ropt=np.ones(d)*0
            
        Tbd_in=x_ropt-(Tbu-Tbd)/4
        Tbu_in=x_ropt+(Tbu-Tbd)/4
        
        
        Xt_ini=init_latin_hypercube_sampling(Tbd_in, Tbu_in, 80,np.random.RandomState(self.r))
        Yt_ini=fo(Xt_ini)           
            
        EV_inf=[]
            
        T1=time.time()



        while N_train<maxN and np.max(np.std(pop,0))>1.0e-6 and Tc==1:

                        
     
            
            pop_child=np.zeros((n,d))
            for i in range(0, n):
                x = optimizer.ask()
  
                pop_child[i,:] = x
            pop1= pop_child 
            pop=pop1



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
             

            
            
            if math.isnan(max(mean)) == False:
                solutions=[]
                for i in range(0, n):
                    x= pop1[i,:]
                    value = fpop[i]
                    solutions.append((x, value))
           
                Ib=np.where(fpop==np.min(fpop))[0][0]
                if fgbest>fpop[Ib]:
                    fgbest=fpop[Ib]
                    gbest=pop[Ib,:]                  
                optimizer.tell(solutions)           

                fpop_real=fo(pop1)
                sorted_id = sorted(range(len(fpop_real)), key=lambda k: fpop_real[k], reverse=False)
                ind_r=np.arange(0, n, 1)
        
                a=(np.array(sorted_id)==ind_r)
                px=np.sum(a==1)/len(a)
                Ac_px=np.append(Ac_px, px)#  当前最优值到真值最优值的欧式距离        
                
                m_ini, var_ini= model.model.predict(Xt_ini)
                sorted_pin = sorted(range(len(m_ini)), key=lambda k: m_ini[k], reverse=False)
                
                sorted_rin = sorted(range(len(Yt_ini)), key=lambda k: Yt_ini[k], reverse=False)
                
        
                a=(np.array(sorted_pin)==np.array(sorted_rin))
                px=np.sum(a==1)/len(a)
                Ac_ini=np.append(Ac_ini, px)#  当前最优值到真值最优值的欧式距离        
                        


                
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












