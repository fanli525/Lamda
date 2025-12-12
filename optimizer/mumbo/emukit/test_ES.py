
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import GPy
import time

### Emukit imports
from emukit.test_functions import forrester_function
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.bayesian_optimization.acquisitions import EntropySearch, ExpectedImprovement, MaxValueEntropySearch
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
LEGEND_SIZE=15
target_function, space = forrester_function()
x_plot = np.linspace(space.parameters[0].min, space.parameters[0].max, 200)[:, None]
y_plot = target_function(x_plot)
X_init = np.array([[0.2],[0.6], [0.9]])
Y_init = target_function(X_init)
plt.figure(figsize=(12, 8))
plt.plot(x_plot, y_plot, "k", label="Objective Function")
plt.scatter(X_init,Y_init)
plt.legend(loc=2, prop={'size': LEGEND_SIZE})
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid(True)
plt.xlim(0, 1)
plt.show()

Y_init=np.atleast_2d(Y_init).reshape(-1,1)
gpy_model = GPy.models.GPRegression(X_init, Y_init, GPy.kern.RBF(1))
emukit_model = GPyModelWrapper(gpy_model)

ei_acquisition = ExpectedImprovement(emukit_model)
es_acquisition = EntropySearch(emukit_model,space)
mes_acquisition = MaxValueEntropySearch(emukit_model,space)
t_0=time.time()
ei_plot = ei_acquisition.evaluate(x_plot)
t_ei=time.time()-t_0
es_plot = es_acquisition.evaluate(x_plot)
t_es=time.time()-t_ei
mes_plot = mes_acquisition.evaluate(x_plot)
t_mes=time.time()-t_es
plt.figure(figsize=(12, 8))
plt.plot(x_plot, (es_plot - np.min(es_plot)) / (np.max(es_plot) - np.min(es_plot)), "green", label="Entropy Search")
plt.plot(x_plot, (ei_plot - np.min(ei_plot)) / (np.max(ei_plot) - np.min(ei_plot)), "blue", label="Expected Improvement")
plt.plot(x_plot, (mes_plot - np.min(mes_plot)) / (np.max(mes_plot) - np.min(mes_plot)), "red", label="Max Value Entropy Search")
plt.legend(loc=1, prop={'size': LEGEND_SIZE})
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid(True)
plt.xlim(0, 1)
plt.show()
plt.figure(figsize=(12, 8))
plt.bar(["ei","es","mes"],[t_ei,t_es,t_mes])
plt.xlabel("Acquisition Choice")
plt.yscale('log')
plt.ylabel("Calculation Time (secs)")






















