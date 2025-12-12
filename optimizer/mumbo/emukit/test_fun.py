# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:54:25 2021

@author: fl347
"""
import numpy as np


import abc
from abc import abstractmethod
class Erastrigin:  
    def __init__(self,d_log,d_no_log):  
        self.d_log=d_log   
        self.d_no_log=d_no_log 

    def __call__(self,x):  
        x_log=x[:,self.d_log]    
        x_no_log=x[:,self.d_no_log]     

        y=1/20*(10.0*x_no_log.shape[1]+np.sum(x_no_log**2-10.0*np.cos(2*np.pi*x_no_log),1))
        y=y+np.sum(np.exp(abs(30-x_log)/100),1)
        return y

class OptimizationProblem(object):
    """Base class for optimization problems."""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.dim = None
        self.lb = None
        self.ub = None
        self.int_var = None
        self.cont_var = None

    def __check_input__(self, x):
        if len(x) != self.dim:
            raise ValueError("Dimension mismatch")

    @abstractmethod
    def eval(self, record):  # pragma: no cover
        pass
    
    
    
def cos_low(x):
    return np.cos(25.*np.sum(x,1))


def nonlinear_sin_low(x, sd=0):
    """
    Low fidelity version of nonlinear sin function
    """

    return np.sin(8 * np.pi * x) + np.random.randn(x.shape[0], 1) * sd


def nonlinear_sin_high(x, sd=0):
    """
    High fidelity version of nonlinear sin function
    """

    return (x - np.sqrt(2)) * nonlinear_sin_low(x, 0) ** 2 + np.random.randn(x.shape[0], 1) * sd



def sp(x):        
    y=np.sum(x**2,1)
    return y





def rastrigin(x):        
    y=1/20*(10.0*x.shape[1]+np.sum(x**2-10.0*np.cos(2*np.pi*x),1))
    # y=y+forrester(x)/1000
    return y

def rosenbrock(x):        
    y=np.sum(100*(x[:,1:]-x[:,0:-1]**2)**2+(x[:,0:-1]-1)**2,1)
   
    return y 


def ackley(x):        
    a =20
    b =0.2
    c =2*np.pi

    y=-a*np.exp(-b*np.sqrt(np.sum(x**2,1)/x.shape[1]))-np.exp(np.sum(np.cos(c*x),1)/x.shape[1])+a+np.exp(1)
       
    return y 

def griewank(x):        
    a=np.linspace(1,x.shape[1],x.shape[1])
    a=np.cos(x/a).prod(axis=1)
    y=np.sum(x**2/4000.0,1)-a+1  
    return y


def forrester(x):
    y =((x*6-2)**2)*np.sin((x*6-2)*2)
    return y[:,0]

def bas_mean(x,xtest,y2):
        
    y=np.sum(xtest**2,1)
    y1m=np.mean(y)
    y1st=np.std(y)
    y2m=np.mean(y2)
    y2st=np.std(y2)
    y=np.sum(x**2,1)
    y1_m=(y-y1m)/y1st
    y1_m=y1_m*y2st+y2m
    
    return y1_m

def bas_min(x,xtest,y2): 
    y=np.sum(xtest**2,1)
       
    y1min=np.min(y)
    y1max=np.max(y)
    y2min=np.min(y2)
    y2max=np.max(y2)
    y=np.sum(x**2,1)

    y1_=(y-y1min)/(y1max-y1min)
    y1_=y1_*(y2max-y2min)+y2min
    
    return y1_





def MFB1(x,phi): 
    n=x.shape[0]
    c=x.shape[1]
    f=c+np.sum(x**2-np.cos(10*np.pi*x),1)
    theta=1.0-0.0001*phi
    a=theta*np.ones((n,c))
    w=10*np.pi*theta*np.ones((n,c))
    b=0.5*np.pi*theta*np.ones((n,c))
    e=np.sum(a*np.cos(w*x+b+np.pi),1)
    obj=f+e
    cost=np.ones((n,1))*phi      
    return  obj,f,e,cost

    
def E_MFB1(x,phi,a0,b0,w0): 
    n=x.shape[0]
    c=x.shape[1]
    theta=1.0-0.0001*phi
    a=theta*np.ones((n,c))

    w=10*np.pi*theta*np.ones((n,c))
    b=0.5*np.pi*theta*np.ones((n,c))
    
    a=a*a0
    b=b*b0
    w=w*w0
    e=np.sum(a*np.cos(w*x+b+np.pi),1)
    return  e


class Exponential(OptimizationProblem):
    """Exponential function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^n e^{jx_j} - \\sum_{j=1} e^{-5.12 j}

    subject to

    .. math::
        -5.12 \\leq x_i \\leq 5.12

    Global optimum: :math:`f(0,0,...,0)=0`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = -5.12 * np.ones(dim)
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Exponential function \n" + "Global optimum: f(-5.12,-5.12,...,-5.12) = 0"

    def __call__(self, X):
        """Evaluate the Exponential function  at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        # self.__check_input__(x)
        n=X.shape[0]
        y=np.zeros(n)
        for j in range(n):
            x=X[j,:]
            total = 0.0
            for i in range(len(x)):
                total += np.exp((i + 1) * x[i - 1]) - np.exp(-5.12 * (i + 1))
            y[j]=total
        return y

class Zakharov(OptimizationProblem):
    """Zakharov function

    Global optimum: :math:`f(0,0,...,0)=1`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0.0
        self.minimum = np.zeros(dim)
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Zakharov function \n" + "Global optimum: f(0,0,...,0) = 1"

    def __call__(self, x):
        """Evaluate the Zakharov function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        # self.__check_input__(x)
        return (
            np.sum(x ** 2,1)
            + np.sum(0.5 * (1 + np.arange(self.dim)) * x,1) ** 2
            + np.sum(0.5 * (1 + np.arange(self.dim)) * x,1) ** 4
        )
    
class Weierstrass(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = np.zeros(dim)
        self.lb = -5 * np.ones(dim)
        self.ub = 5 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Weierstrass function"

    def __call__(self, X):
        """Evaluate the Weierstrass function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        # self.__check_input__(x)
        d = X.shape[1]
        n=X.shape[0]
        y=np.zeros(n)
        for j in range(n):
            x=X[j,:]

            f0, val = 0.0, 0.0
            for k in range(12):
                f0 += 1.0 / (2 ** k) * np.cos(np.pi * (3 ** k))
                for i in range(d):
                    val += 1.0 / (2 ** k) * np.cos(2 * np.pi * (3 ** k) * (x[i] + 0.5))
            y[j] =10 * ((1.0 / float(d) * val - f0) ** 3)
        return y
   


class Himmelblau(OptimizationProblem):
    """Himmelblau function

    .. math::
        f(x_1,\\ldots,x_n) = 10n -
            \\frac{1}{2n} \\sum_{i=1}^n (x_i^4 - 16x_i^2 + 5x_i)

    Global optimum: :math:`f(-2.903,...,-2.903)=-39.166`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = -39.166165703771412
        self.minimum = -2.903534027771178 * np.ones(dim)
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Himmelblau function \n" + "Global optimum: f(-2.903,...,-2.903) = -39.166"

    def __call__(self, x):
        """Evaluate the Himmelblau function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        # self.__check_input__(x)
        return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x,1) / float(self.dim)


class Levy(OptimizationProblem):
    """Levy function

    Details: https://www.sfu.ca/~ssurjano/levy.html

    Global optimum: :math:`f(1,1,...,1)=0`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0.0
        self.minimum = np.ones(dim)
        self.lb = -5 * np.ones(dim)
        self.ub = 5 * np.ones(dim)
        # 与网上边界不同，网上-10 10
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Levy function \n" + "Global optimum: f(1,1,...,1) = 0"

    def __call__(self, x):
        """Evaluate the Levy function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        # self.__check_input__(x)
        w = 1 + (x - 1.0) / 4.0
        d = self.dim
        # return (
        #     np.sin(np.pi * w[:,0]) ** 2
        #     + np.sum((w[:,1 : d - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:,1 : d - 1] + 1) ** 2),1)
        #     + (w[:,d - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:,d - 1]) ** 2)
        # )
    
        return (
            np.sin(np.pi * w[:,0]) ** 2
            + np.sum((w[:,0 : d - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:,0 : d - 1] + 1) ** 2),1)
            + (w[:,d - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:,d - 1]) ** 2)
        )
        
    
class Michalewicz(OptimizationProblem):
    """Michalewicz function

    .. math::
        f(x_1,\\ldots,x_n) = -\\sum_{i=1}^n \\sin(x_i) \\sin^{20}
            \\left( \\frac{ix_i^2}{\\pi} \\right)

    subject to

    .. math::
        0 \\leq x_i \\leq \\pi

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.lb = np.zeros(dim)
        self.ub = np.pi * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Michalewicz function \n" + "Global optimum: ??"

    def __call__(self, x):
        """Evaluate the Michalewicz function  at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        # self.__check_input__(x)
        return -np.sum(np.sin(x) * (np.sin(((1 + np.arange(self.dim)) * x ** 2) / np.pi)) ** 20,1)



class Perm(OptimizationProblem):
    """Perm function

    Global optimum: :math:`f(1,1/2,1/3,...,1/n)=0`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0.0
        self.minimum = np.ones(dim) / np.arange(1, dim + 1)
        self.lb = -5 * np.ones(dim)
        self.ub = 5 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Perm function \n" + "Global optimum: f(1,1/2,1/3...,1/d) = 0"

    def __call__(self,X):
        """Evaluate the Perm function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        # self.__check_input__(x)
        beta = 10.0
        d = X.shape[1]
        n=X.shape[0]
        y=np.zeros(n)
        for k in range(n):
            x=X[k,:] 
            outer = 0.0
            for ii in range(d):
                inner = 0.0
                for jj in range(d):
                    xj = x[jj]
                    inner += ((jj + 1) + beta) * (xj ** (ii + 1) - (1.0 / (jj + 1)) ** (ii + 1))
                outer += inner ** 2
            y[k]=outer
        return y


class Schwefel(OptimizationProblem):
    """Schwefel function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^{n} \
        \\left( -x_j \\sin(\\sqrt{|x_j|}) \\right) + 418.982997 n

    subject to

    .. math::
        -512 \\leq x_i \\leq 512

    Global optimum: :math:`f(420.968746,420.968746,...,420.968746)=0`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = 420.968746 * np.ones(dim)
        self.lb = -512 * np.ones(dim)
        self.ub = 512 * np.ones(dim)
        # 网上边界-500 500
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Schwefel function \n" + "Global optimum: f(420.9687,...,420.9687) = 0"

    def __call__(self, X):
        """Evaluate the Schwefel function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        # self.__check_input__(x)

        n=X.shape[0]
        y1=np.zeros(n)
        for k in range(n):
            x=X[k,:]
            y1[k]=418.9829 * self.dim - sum([y * np.sin(np.sqrt(abs(y))) for y in x])
        
        
        return y1

class Schwefel_rist(OptimizationProblem):
    """Schwefel function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^{n} \
        \\left( -x_j \\sin(\\sqrt{|x_j|}) \\right) + 418.982997 n

    subject to

    .. math::
        -512 \\leq x_i \\leq 512

    Global optimum: :math:`f(420.968746,420.968746,...,420.968746)=0`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = 420.968746 * np.ones(dim)
        self.lb = -512 * np.ones(dim)
        self.ub = 512 * np.ones(dim)
        # 网上边界-500 500
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Schwefel function \n" + "Global optimum: f(420.9687,...,420.9687) = 0"

    def __call__(self, X):
        """Evaluate the Schwefel function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        # self.__check_input__(x)

        n=X.shape[0]
        y1=np.zeros(n)
        for k in range(n):
            x=X[k,:]
            y1[k]=sum([y * np.sin(np.sqrt(abs(y))) for y in x])
        
        y=rastrigin(X)
        return y1+y
    
    
class Michalewicz_rist(OptimizationProblem):
    """Michalewicz function

    .. math::
        f(x_1,\\ldots,x_n) = -\\sum_{i=1}^n \\sin(x_i) \\sin^{20}
            \\left( \\frac{ix_i^2}{\\pi} \\right)

    subject to

    .. math::
        0 \\leq x_i \\leq \\pi

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.lb = np.zeros(dim)
        self.ub = np.pi * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Michalewicz function \n" + "Global optimum: ??"

    def __call__(self, x):
        """Evaluate the Michalewicz function  at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        # self.__check_input__(x)
        return rastrigin(x)-np.sum(np.sin(x) * (np.sin(((1 + np.arange(self.dim)) * x ** 2) / np.pi)) ** 20,1)

