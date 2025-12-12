import numpy as np
import pylab as pb
import GPy
def f(X):
    return 10. + .1*X + 2*np.sin(X)/X

fig,ax = pb.subplots()
ax.plot(np.linspace(-15,25),f(np.linspace(-10,20)),'r-')
ax.grid()
X = np.random.uniform(-10,20, 50)
X = X[~np.logical_and(X>-2,X<3)] #Remove points between -2 and 3 (just for illustration)
X = np.hstack([np.random.uniform(-1,1,1),X]) #Prepend a point between -1 and 1  (just for illustration)
error = np.random.normal(0,.2,X.size)
Y = f(X) + error
fig,ax = pb.subplots()
ax.plot(np.linspace(-15,25),f(np.linspace(-10,20)),'r-')
ax.plot(X,Y,'kx',mew=1.5)
ax.grid()
kern = GPy.kern.MLP(1) + GPy.kern.Bias(1)
m = GPy.models.GPHeteroscedasticRegression(X[:,None],Y[:,None],kern)
# m.het_Gauss.variance.fix(0.0)


# m['.*het_Gauss.variance'] = abs(error)[:,None] #Set the noise parameters to the error in Y
# m.het_Gauss.variance.fix() #We can fix the noise term, since we already know it
m.optimize()
mu, var = m._raw_predict(m.X)

m.predict(X.reshape(-1,1))
m.plot_f() #Show the predictive values of the GP.
# pb.errorbar(X,Y,yerr=np.array(m.likelihood.flattened_parameters).flatten(),fmt='',ecolor='r',zorder=1)
pb.grid()
pb.plot(X,Y,'kx',mew=1.5)
pb.show()


def noise_effect(noise):
    m.het_Gauss.variance[:1] = noise
    m.het_Gauss.variance.fix()
    m.optimize()

    m.plot_f()
    pb.errorbar(X.flatten(), Y.flatten(), yerr=np.array(m.likelihood.flattened_parameters).flatten(), fmt=None,
                ecolor='r', zorder=1)
    pb.plot(X[1:], Y[1:], 'kx', mew=1.5)
    pb.plot(X[:1], Y[:1], 'ko', mew=.5)
    pb.grid()

# from IPython.html.widgets import *
# interact(noise_effect, noise=(0.1,2.))


m1 = GPy.models.GPHeteroscedasticRegression(X[:,None],Y[:,None],kern)
m1.het_Gauss.variance = .05
m1.het_Gauss.variance.fix()
m1.optimize()

# Homoscedastic model
m2 = GPy.models.GPRegression(X[:,None],Y[:,None],kern)
m2['.*Gaussian_noise'] = .05
m2['.*noise'].fix()
m2.optimize()

m1.plot_f()
pb.title('Homoscedastic model')
m2.plot_f()
pb.title('Heteroscedastic model')
pb.show()

print("Kernel parameters (optimized) in the heteroscedastic model")
print(m1.kern)
print ("\nKernel parameters (optimized) in the homoscedastic model")
print( m2.kern)