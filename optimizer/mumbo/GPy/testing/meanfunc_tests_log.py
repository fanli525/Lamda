# Copyright (c) 2015, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

def fun(X):
    Y = 1+np.exp(-0.1*X)
    return Y

import matplotlib.pyplot as plt
lb=1
ub=100
Nt=3

X = np.linspace(lb,ub,Nt).reshape(-1,1)
X = np.random.uniform(lb,ub, size=(Nt, 1))
X[0,0]=1
Y = fun(X)
Xt = np.linspace(lb,ub,500).reshape(-1,1)
Yt = fun(Xt)
mf = GPy.core.Mapping(1,1)

# def mean_func(X):
#      # return a1 * np.power(a2, a3*t) + a4
#     return np.mean(Y)
#
# mf.f = mean_func
# mf.update_gradients = lambda a,b: None
# k =GPy.kern.Linear(1)
# k =GPy.kern.RBF(1)
# lik = GPy.likelihoods.Gaussian()
# m = GPy.core.GP(X,Y, kernel=k, likelihood=lik,mean_function=mf)
# m.Gaussian_noise.fix(1.0e-8)
# # plt.plot(Xt , Yt  , 'k--',label='real')
# # plt.plot(X, Y  , 'ro',label='Tra')
# # plt.legend(loc='upper left')
# # plt.xlabel('runtime')
# # plt.show()
#
# m.optimize()
# pre,var=m.predict(Xt)
# sigma=np.sqrt(var)
# plt.plot(Xt , Yt  , 'k--',label='real')
# plt.plot(Xt , pre  , 'g-',label='pre')
# plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
#                  (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
# plt.plot(X, Y , 'ro',label='Tra')
# plt.legend(loc='upper left')
# plt.xlabel('runtime')
# plt.title('meanf=mean(Y) Nt='+str(Nt))
# plt.show()
#
#
#
# mf = GPy.core.Mapping(1,1)
# def mean_func1(X):
#     # return a1 * np.power(a2, a3*t) + a4
#     return np.exp(-1*X)
# mf.f = mean_func1
# mf.update_gradients = lambda a,b: None
# k =GPy.kern.Linear(1)
# k =GPy.kern.RBF(1)
#
# lik = GPy.likelihoods.Gaussian()
# m = GPy.core.GP(X, Y-mean_func1(X), kernel=k, likelihood=lik)
# m.Gaussian_noise.fix(1.0e-8)
#
# m.optimize()
# pre,var=m.predict(Xt)
# sigma=np.sqrt(var)
# plt.plot(Xt , Yt  , 'k--',label='real')
# plt.plot(Xt , mean_func1(Xt) , 'b-',label='mean')
#
# plt.plot(Xt , pre+mean_func1(Xt) , 'g-',label='pre')
# plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
#                  (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
# plt.plot(X, Y , 'ro',label='Tra')
# plt.legend(loc='upper left')
# plt.xlabel('runtime')
# plt.title('meanf=np.exp(-1*X) Nt='+str(Nt))
#
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
# import scipy
# fit_bounds = ([-max(Y), 0, -np.inf], [max(Y), np.inf, np.inf])
# popt, pcov = scipy.optimize.curve_fit(lambda t, a1, a2, a3: a1 * np.log(a2 * t) + a3,
#                                       np.asarray(X+1.0e-8).reshape(-1), np.asarray(Y).reshape(-1)
#                                       , bounds=fit_bounds)
# # popt, pcov = scipy.optimize.curve_fit(lambda t, a1, a2, a3, a4: a1 * np.power(a2, a3*t) + a4,
# #                                       np.asarray(X_index).reshape(-1), np.asarray(Y_best).reshape(-1), bounds=fit_bounds)
# # mf.f = np.log
# a1, a2, a3 = popt
#
# def mean_func(t):
#     # return a1 * np.power(a2, a3*t) + a4
#     return a1 * np.log(a2 * t) + a3
#
# mf.f = mean_func
# mf.update_gradients = lambda a, b: None
# k1 = GPy.kern.Matern52(1)
# k2 = GPy.kern.Linear(1)
# k=GPy.kern.Add([k1,k2])
#
# lik = GPy.likelihoods.Gaussian()
# m = GPy.core.GP(X, Y, kernel=k1, likelihood=lik, mean_function=mf)
# m.optimize()
# m.Gaussian_noise.fix(1.0e-8)
#
#
# pre,var=m.predict(Xt)
# sigma=np.sqrt(var)
# plt.plot(Xt , Yt  , 'k--',label='real')
# plt.plot(Xt , mean_func(Xt) , 'b-',label='prior mean')
# plt.plot(Xt , pre  , 'g-',label='pre')
# plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
#                  (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
# plt.plot(X, Y , 'ro',label='Tra')
# plt.legend(loc='upper left')
# plt.xlabel('runtime')
# plt.title('meanf=clog(ax)+b Nt='+str(Nt))
# plt.show()
#
#
# k = GPy.kern.Matern52(1)
# # k = GPy.kern.Linear(1)
# lik = GPy.likelihoods.Gaussian()
# m = GPy.core.GP(X, Y-mean_func(X), kernel=k, likelihood=lik)
# m.optimize()
# m.Gaussian_noise.fix(1.0e-8)
# pre,var=m.predict(Xt)
# pre=pre+mean_func(Xt)
# sigma=np.sqrt(var)
# plt.plot(Xt , Yt  , 'k--',label='real')
# plt.plot(Xt , mean_func(Xt) , 'b-',label='mean')
# plt.plot(Xt , pre  , 'g-',label='pre')
# plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
#                  (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
# plt.plot(X, Y , 'ro',label='Tra')
# plt.legend(loc='upper left')
# plt.xlabel('runtime')
# plt.title('meanf=clog(ax)+b Nt='+str(Nt))
# plt.show()
#
#
#
#
# k = GPy.kern.Matern52(1)
# # k = GPy.kern.Linear(1)
# lik = GPy.likelihoods.Gaussian()
# m = GPy.models.GPRegression(X, Y-mean_func(X), kernel=k)
# m.optimize()
# m.Gaussian_noise.fix(1.0e-8)
# pre,var=m.predict(Xt)
# sigma=np.sqrt(var)
# plt.plot(Xt , Yt-mean_func(Xt) , 'k--',label='real')
# plt.plot(Xt , pre  , 'g-',label='pre')
# plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
#                  (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
# plt.plot(X, Y-mean_func(X) , 'ro',label='Tra')
# plt.legend(loc='upper left')
# plt.xlabel('runtime')
# plt.title('meanf=clog(ax)+b error Nt='+str(Nt))
# plt.show()
#
#
# def mean_func(X):
#     # return a1 * np.power(a2, a3*t) + a4
#     return np.mean(np.log(Y))
#
#

k = GPy.kern.Linear(1)
lik = GPy.likelihoods.Gaussian()
m = GPy.core.GP(X, np.log(Y),kernel=k,likelihood=lik)
m.optimize()
m.Gaussian_noise.fix(1.0e-8)
pre,var=m.predict(Xt)
pre=np.exp(pre)
sigma=np.sqrt(var)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper right')
plt.xlabel('runtime')
plt.title('L log(y) Nt='+str(Nt))
plt.show()


k = GPy.kern.Linear(1)
lik = GPy.likelihoods.Gaussian()
m = GPy.core.GP(X, Y,kernel=k,likelihood=lik)
m.optimize()
m.Gaussian_noise.fix(1.0e-8)
pre,var=m.predict(Xt)
pre=np.exp(pre)
sigma=np.sqrt(var)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper right')
plt.xlabel('runtime')
plt.title('L  Nt='+str(Nt))
plt.show()




k = GPy.kern.RatQuad(1)
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
plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper right')
plt.xlabel('runtime')
plt.title('RatQuad log(y) Nt='+str(Nt))
plt.show()

k = GPy.kern.RBF(1)
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
plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper right')
plt.xlabel('runtime')
plt.title('RBF log(y) Nt='+str(Nt))
plt.show()

k = GPy.kern.MLP(1)
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
plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper right')
plt.xlabel('runtime')
plt.title('MLP log(y) Nt='+str(Nt))
plt.show()

k1 = GPy.kern.RatQuad(1)
k2 = GPy.kern.Linear(1)
K_add = GPy.kern.Add([k1, k2])
K_pro = GPy.kern.Prod([k1, k2])
K = GPy.kern.Add([K_add, K_pro])


lik = GPy.likelihoods.Gaussian()
m = GPy.models.GPRegression(X, np.log(Y),kernel=K)
m.optimize()
m.Gaussian_noise.fix(1.0e-8)
pre,var=m.predict(Xt)
pre=np.exp(pre)
sigma=np.sqrt(var)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper right')
plt.xlabel('runtime')
plt.title('RQ+Li log(y) Nt='+str(Nt))
plt.show()



k1 = GPy.kern.MLP(1)
k2 = GPy.kern.RatQuad(1)
K_add = GPy.kern.Add([k1, k2])
K_pro = GPy.kern.Prod([k1, k2])
K = GPy.kern.Add([K_add, K_pro])


lik = GPy.likelihoods.Gaussian()
m = GPy.models.GPRegression(X, np.log(Y),kernel=K)
m.optimize()
m.Gaussian_noise.fix(1.0e-8)
pre,var=m.predict(Xt)
pre=np.exp(pre)
sigma=np.sqrt(var)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper right')
plt.xlabel('runtime')
plt.title('MLP+RQ log(y) Nt='+str(Nt))
plt.show()


k1 = GPy.kern.MLP(1)
k2 = GPy.kern.Linear(1)
K_add = GPy.kern.Add([k1, k2])
K_pro = GPy.kern.Prod([k1, k2])
K = GPy.kern.Add([K_add, K_pro])


lik = GPy.likelihoods.Gaussian()
m = GPy.models.GPRegression(X, np.log(Y),kernel=K)
m.optimize()
m.Gaussian_noise.fix(1.0e-8)
pre,var=m.predict(Xt)
pre=np.exp(pre)
sigma=np.sqrt(var)
plt.plot(Xt , Yt  , 'k--',label='real')
plt.plot(Xt , pre  , 'g-',label='pre')
plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
                 (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
plt.plot(X, Y  , 'ro',label='Tra')
plt.legend(loc='upper right')
plt.xlabel('runtime')
plt.title('MLP+Li log(y) Nt='+str(Nt))
plt.show()




# from GPy.models.warped_gp import WarpedGP
# k = GPy.kern.RBF(1)
# m = WarpedGP(X, Y,kernel=k)
# m.optimize()
# m.Gaussian_noise.fix(1.0e-8)
# pre,var=m.predict(Xt)
# plt.plot(Xt , Yt  , 'k--',label='real')
# plt.plot(Xt , pre  , 'g-',label='pre')
# plt.fill_between(Xt[:,0], (pre[:,0] - 1 * sigma[:,0] ).flatten(),
#                  (pre[:,0]  + 1 * sigma[:,0] ).flatten(), facecolor='b', alpha=0.3)
# plt.plot(X, Y  , 'ro',label='Tra')
# plt.legend(loc='upper right')
# plt.xlabel('runtime')
# plt.title('output warp Nt='+str(Nt))
# plt.show()
#
#

n =300
x =np.linspace(lb,ub,n).reshape(-1,1)

k1 = GPy.kern.MLP(1)
# k1 = GPy.kern.RBF(1)

C = k1.K(x,x)
u, s,  vh= np.linalg.svd(C, full_matrices=True)
t=10
z_gp = np.zeros((t,n))
s=np.diag(s)
for i in range(t):
    gn = np.random.randn(n)
    z_gp[i,:] = np.dot(np.dot(u ,np.sqrt(s)),gn)
    plt.plot(x, z_gp[i,:])


plt.title('MLP')
plt.show()


n =300
x =np.linspace(lb,ub,n).reshape(-1,1)

k1 = GPy.kern.Linear(1)
# k1 = GPy.kern.RBF(1)

C = k1.K(x,x)
u, s,  vh= np.linalg.svd(C, full_matrices=True)
t=10
z_gp = np.zeros((t,n))
s=np.diag(s)
for i in range(t):
    gn = np.random.randn(n)
    z_gp[i,:] = np.dot(np.dot(u ,np.sqrt(s)),gn)
    plt.plot(x, z_gp[i,:])


plt.title('Linear')
plt.show()


n =300
x =np.linspace(lb,ub,n).reshape(-1,1)

k1 = GPy.kern.MLP(1)
k2 = GPy.kern.Linear(1)
K_add = GPy.kern.Add([k1, k2])
K_pro = GPy.kern.Prod([k1, k2])
K = GPy.kern.Add([K_add, K_pro])

C = K.K(x,x)
u, s,  vh= np.linalg.svd(C, full_matrices=True)
t=10
z_gp = np.zeros((t,n))
s=np.diag(s)
for i in range(t):
    gn = np.random.randn(n)
    z_gp[i,:] = np.dot(np.dot(u ,np.sqrt(s)),gn)
    plt.plot(x, z_gp[i,:])


plt.title('MLP+Linear')
plt.show()





n =300
x =np.linspace(lb,ub,n).reshape(-1,1)

k1 = GPy.kern.MLP(1)
k2 = GPy.kern.RatQuad(1)
K_add = GPy.kern.Add([k1, k2])
K_pro = GPy.kern.Prod([k1, k2])
K = GPy.kern.Add([K_add, K_pro])

C = K.K(x,x)
u, s,  vh= np.linalg.svd(C, full_matrices=True)
t=10
z_gp = np.zeros((t,n))
s=np.diag(s)
for i in range(t):
    gn = np.random.randn(n)
    z_gp[i,:] = np.dot(np.dot(u ,np.sqrt(s)),gn)
    plt.plot(x, z_gp[i,:])


plt.title('MLP+RatQuad')
plt.show()

