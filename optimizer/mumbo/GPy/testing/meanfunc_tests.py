# Copyright (c) 2015, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class MFtests(unittest.TestCase):
    def test_simple_mean_function(self):
        """
        The simplest possible mean function. No parameters, just a simple Sinusoid.
        """
        #create  simple mean function
        mf = GPy.core.Mapping(1,1)
        mf.f = np.sin
        mf.update_gradients = lambda a,b: None

        X = np.linspace(0,10,50).reshape(-1,1)
        Y = np.sin(X) + 0.5*np.cos(3*X) + 0.1*np.random.randn(*X.shape)

        k =GPy.kern.RBF(1)
        lik = GPy.likelihoods.Gaussian()
        m = GPy.core.GP(X, Y, kernel=k, likelihood=lik, mean_function=mf)
        self.assertTrue(m.checkgrad())

    def test_parametric_mean_function(self):
        """
        A linear mean function with parameters that we'll learn alongside the kernel
        """

        X = np.linspace(-1,10,50).reshape(-1,1)
        
        Y = 3-np.abs((X-6))
        Y += .5*np.cos(3*X) + 0.3*np.random.randn(*X.shape) 

        mf = GPy.mappings.PiecewiseLinear(1, 1, [-1,1], [9,2])

        k =GPy.kern.RBF(1)
        lik = GPy.likelihoods.Gaussian()
        m = GPy.core.GP(X, Y, kernel=k, likelihood=lik, mean_function=mf)
        self.assertTrue(m.checkgrad())

    def test_parametric_mean_function_composition(self):
        """
        A linear mean function with parameters that we'll learn alongside the kernel
        """

        X = np.linspace(0,10,50).reshape(-1,1)
        Y = np.sin(X) + 0.5*np.cos(3*X) + 0.1*np.random.randn(*X.shape) + 3*X

        mf = GPy.mappings.Compound(GPy.mappings.Linear(1,1), 
                                   GPy.mappings.Kernel(1, 1, np.random.normal(0,1,(1,1)), 
                                                       GPy.kern.RBF(1))
                                   )

        k =GPy.kern.RBF(1)
        lik = GPy.likelihoods.Gaussian()
        m = GPy.core.GP(X, Y, kernel=k, likelihood=lik, mean_function=mf)
        self.assertTrue(m.checkgrad())

    def test_parametric_mean_function_additive(self):
        """
        A linear mean function with parameters that we'll learn alongside the kernel
        """

        X = np.linspace(0,10,50).reshape(-1,1)
        Y = np.sin(X) + 0.5*np.cos(3*X) + 0.1*np.random.randn(*X.shape) + 3*X

        mf = GPy.mappings.Additive(GPy.mappings.Constant(1,1,3),
               GPy.mappings.Additive(GPy.mappings.MLP(1,1),
                     GPy.mappings.Identity(1,1)
                           )
                        )

        k =GPy.kern.RBF(1)
        lik = GPy.likelihoods.Gaussian()
        m = GPy.core.GP(X, Y, kernel=k, likelihood=lik, mean_function=mf)
        self.assertTrue(m.checkgrad())

    def test_svgp_mean_function(self):

        # an instance of the SVIGOP with a men function
        X = np.linspace(0,10,500).reshape(-1,1)
        Y = np.sin(X) + 0.5*np.cos(3*X) + 0.1*np.random.randn(*X.shape)
        Y = np.where(Y>0, 1,0) # make aclassificatino problem

        mf = GPy.mappings.Linear(1,1)
        Z = np.linspace(0,10,50).reshape(-1,1)
        lik = GPy.likelihoods.Bernoulli()
        k =GPy.kern.RBF(1) + GPy.kern.White(1, 1e-4)
        m = GPy.core.SVGP(X, Y,Z=Z, kernel=k, likelihood=lik, mean_function=mf)
        self.assertTrue(m.checkgrad())


def fun(X):
    Y = 1+np.exp(-1*X)
    return Y

import matplotlib.pyplot as plt
lb=1
ub=100
Nt=6

X = np.linspace(lb,ub,Nt).reshape(-1,1)
Y = fun(X)
Xt = np.linspace(lb,ub,500).reshape(-1,1)
Yt = fun(Xt)
mf = GPy.core.Mapping(1,1)

def mean_func(X):
     # return a1 * np.power(a2, a3*t) + a4
    return np.mean(Y)

mf.f = mean_func
mf.update_gradients = lambda a,b: None
k =GPy.kern.Linear(1)
k =GPy.kern.RBF(1)
lik = GPy.likelihoods.Gaussian()
m = GPy.core.GP(X,Y, kernel=k, likelihood=lik,mean_function=mf)
m.Gaussian_noise.fix(1.0e-8)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper left')
plt.xlabel('runtime')
plt.show()

m.optimize()
pre,var=m.predict(Xt)
sigma=np.sqrt(var)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 0.2 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 0.2 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y , 'ro',label='Tra')
plt.legend(loc='upper left')
plt.xlabel('runtime')
plt.show()



mf = GPy.core.Mapping(1,1)
def mean_func1(X):
    # return a1 * np.power(a2, a3*t) + a4
    return np.exp(-1*X)
mf.f = mean_func1
mf.update_gradients = lambda a,b: None
k =GPy.kern.Linear(1)
k =GPy.kern.RBF(1)

lik = GPy.likelihoods.Gaussian()
m = GPy.core.GP(X, Y-mean_func1(X), kernel=k, likelihood=lik,mean_function=mf)
m.Gaussian_noise.fix(1.0e-8)

m.optimize()
pre,var=m.predict(Xt)
sigma=np.sqrt(var)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , mean_func1(Xt) , 'b-',label='mean')

plt.plot(Xt , pre+mean_func1(Xt) , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 0.2 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 0.2 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y , 'ro',label='Tra')
plt.legend(loc='upper left')
plt.xlabel('runtime')
plt.show()












import scipy
fit_bounds = ([-max(Y), 0, -np.inf], [max(Y), np.inf, np.inf])
popt, pcov = scipy.optimize.curve_fit(lambda t, a1, a2, a3: a1 * np.log(a2 * t) + a3,
                                      np.asarray(X+1.0e-8).reshape(-1), np.asarray(Y).reshape(-1)
                                      , bounds=fit_bounds)
# popt, pcov = scipy.optimize.curve_fit(lambda t, a1, a2, a3, a4: a1 * np.power(a2, a3*t) + a4,
#                                       np.asarray(X_index).reshape(-1), np.asarray(Y_best).reshape(-1), bounds=fit_bounds)
# mf.f = np.log
a1, a2, a3 = popt

def mean_func(t):
    # return a1 * np.power(a2, a3*t) + a4
    return a1 * np.log(a2 * t) + a3

mf.f = mean_func
mf.update_gradients = lambda a, b: None
k1 = GPy.kern.Matern52(1)
k2 = GPy.kern.Linear(1)
k=GPy.kern.Add([k1,k2])

lik = GPy.likelihoods.Gaussian()
m = GPy.core.GP(X, Y, kernel=k1, likelihood=lik, mean_function=mf)
m.optimize()
m.Gaussian_noise.fix(1.0e-8)

pre,var=m.predict(Xt)
sigma=np.sqrt(var)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , mean_func(Xt) , 'b-',label='prior mean')

plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 0.2 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 0.2 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y , 'ro',label='Tra')
plt.legend(loc='upper left')
plt.xlabel('runtime')
plt.show()


k = GPy.kern.Matern52(1)
# k = GPy.kern.Linear(1)
lik = GPy.likelihoods.Gaussian()
m = GPy.core.GP(X, Y-mean_func(X), kernel=k, likelihood=lik)
m.optimize()
m.Gaussian_noise.fix(1.0e-8)
pre,var=m.predict(Xt)
pre=pre+mean_func(Xt)
sigma=np.sqrt(var)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , mean_func(Xt) , 'b-',label='mean')
plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 0.2 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 0.2 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y , 'ro',label='Tra')
plt.legend(loc='upper left')
plt.xlabel('runtime')
plt.show()




k = GPy.kern.Matern52(1)
# k = GPy.kern.Linear(1)
lik = GPy.likelihoods.Gaussian()
m = GPy.models.GPRegression(X, Y-mean_func(X), kernel=k)
m.optimize()
m.Gaussian_noise.fix(1.0e-8)
pre,var=m.predict(Xt)
sigma=np.sqrt(var)
plt.plot(Xt , Yt-mean_func(Xt) , 'k--',label='real')
plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 0.2 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 0.2 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y-mean_func(X) , 'ro',label='Tra')
plt.legend(loc='upper left')
plt.xlabel('runtime')
plt.show()









k = GPy.kern.Matern52(1)
# k = GPy.kern.Linear(1)
lik = GPy.likelihoods.Gaussian()
m = GPy.core.GP(X, np.log(Y),kernel=k,likelihood=lik)
m.optimize()
m.Gaussian_noise.fix(1.0e-8)
pre,var=m.predict(Xt)
pre=np.exp(pre)
sigma=np.sqrt(var)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 0.2 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 0.2 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper right')
plt.xlabel('runtime')
plt.title('log(y)')
plt.show()


k = GPy.kern.Matern52(1)
# k = GPy.kern.Linear(1)
lik = GPy.likelihoods.Gaussian()
m = GPy.models.GPRegression(X, np.log(Y),kernel=k)
m.optimize()
m.Gaussian_noise.fix(1.0e-8)
pre,var=m.predict(Xt)
pre=np.exp(pre)
sigma=np.sqrt(var)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 0.2 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 0.2 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper right')
plt.xlabel('runtime')
plt.title('log(y)')
plt.show()




from GPy.models.warped_gp import WarpedGP
k = GPy.kern.Matern52(1)
m = WarpedGP(X, Y,kernel=k)
m.optimize()
m.Gaussian_noise.fix(1.0e-8)
pre,var=m.predict(Xt)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 0.2 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 0.2 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper right')
plt.xlabel('runtime')
plt.title('output warp')
plt.show()



