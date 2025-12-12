

import numpy as np
from scipy.stats import norm
class MF_sPI:
    def __init__(self, FM_name,model, next_task):
        self.model = model
        self.next_task = next_task
        self.FM_name = FM_name

    def __call__(self, x):
        if self.model.n_outputs==1:
            X = x.reshape(1,-1)
        else:
            X=np.zeros((1,len(x)+1))
            X[0,:-1]=x
            X[0,-1]=self.next_task
        py, var = self.model.predict(X)
        sigma = np.sqrt(abs(var))
        y = self.model.Y[self.model.X[:, -1] == self.next_task]
        fmin = min(y)
        Z = (fmin - py) / sigma
        poi= norm.cdf(Z)
        self.poi = -poi
        self.AC_value = poi
        self.sigma = sigma
        return -self.poi[0]


class MF_sEI:
    def __init__(self,  FM_name,model, next_task):
        self.model = model
        self.next_task = next_task
        self.FM_name = FM_name

    def __call__(self, x):
        if self.FM_name== 'SGP':
            X = x.reshape(1,-1)
        else:
            X=np.zeros((1,len(x)+1))
            X[0,:-1]=x
            X[0,-1]=self.next_task

        py, var = self.model.predict(X)
        sigma = np.sqrt(abs(var))
        y = self.model.Y
        fmin = min(y)
        Z = (fmin - py) / sigma
        ei = (fmin - py) * norm.cdf(Z) + sigma * norm.pdf(Z)
        self.ei = ei
        self.AC_value = -ei
        self.sigma = sigma
        return -self.ei[0]



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

class MF_sMEAN:
    def __init__(self,  FM_name,model, next_task):
        self.model = model
        self.next_task = next_task
        self.FM_name = FM_name

    def __call__(self, x):
        if self.model.n_outputs==1:
            X = x.reshape(1,-1)
        else:
            X=np.zeros((1,len(x)+1))
            X[0,:-1]=x
            X[0,-1]=self.next_task
        py, var = self.model.predict(X)
        sigma = np.sqrt(abs(var))

        self.mean = py
        self.AC_value = py
        self.sigma = sigma
        return py[0]


class MF_sVar:
    def __init__(self, FM_name, model, next_task):
        self.model = model
        self.next_task = next_task
        self.FM_name = FM_name

    def __call__(self, x):
        if self.FM_name== 'SGP':
            X = x.reshape(1,-1)
        else:
            X=np.zeros((1,len(x)+1))
            X[0,:-1]=x
            X[0,-1]=self.next_task

        py, var = self.model.predict(X)
        sigma = np.sqrt(abs(var))
        self.sigma = sigma
        self.AC_value = -sigma
        return -sigma[0]


class MEI_cost:
    def __init__(self, model,modelc):
        self.model = model
        self.modelc=modelc
    def __call__(self, x):
        X = x.reshape(1,-1)
        cu_fid=int(X[:, -1] )
        X[:, -1] =cu_fid
        py, var = self.model.predict(X)
        sigma = np.sqrt(abs(var))
        pre_cost,v = self.modelc.predict(X)
        y = self.model.Y
        fmin = min(y)
        Z = (fmin - py) / sigma
        ei = (fmin - py) * norm.cdf(Z) + sigma * norm.pdf(Z)
        self.ei = ei
        self.AC_value = -ei
        self.sigma = sigma
        # return -self.ei[0],-pre_cost[0]
        return np.column_stack([-self.ei[0],-pre_cost[0]])

class MPI_cost:
    def __init__(self, model,cost):
        self.model = model
        self.cost=cost
    def __call__(self, x):
        X = x.reshape(1,-1)
        cu_fid=int(X[:, -1] )
        X[:, -1] =cu_fid
        py, var = self.model.predict(X)
        sigma = np.sqrt(abs(var))
        y = self.model.Y[ self.model.X[:,-1]==cu_fid]
        fmin = min(y)
        Z = (fmin - py) / sigma
        poi= norm.cdf(Z)
        self.poi = poi/self.cost[cu_fid]
        self.AC_value = -poi
        self.sigma = sigma
        return -self.poi[0]

class MVar_cost:
    def __init__(self, model,cost):
        self.model = model
        self.cost=cost
    def __call__(self, x):
        X = x.reshape(1,-1)
        cu_fid=int(X[:, -1] )
        X[:, -1] =cu_fid
        py, var = self.model.predict(X)
        sigma = np.sqrt(abs(var))
        self.AC_value = sigma/self.cost[cu_fid]
        self.sigma = sigma/self.cost[cu_fid]
        return -self.sigma[0]



class MLCB_cost:
    def __init__(self, model,cost,w=-2):
        self.model = model
        self.cost=cost
        self.w = w

    def __call__(self, x):
        X = x.reshape(1,-1)
        cu_fid=int(X[:, -1] )
        X[:, -1] =cu_fid
        py, var = self.model.predict(X)
        sigma = np.sqrt(abs(var))
        self.sigma = sigma
        if np.min(sigma) > 0:
            lcb = py - self.w * sigma
        else:
            lcb = py
        lcb=np.log(1+np.exp(lcb))/self.cost[cu_fid]
        self.AC_value =lcb
        self.lcb =lcb
        return lcb[0]


class MMEAN_cost:
    def __init__(self, model,cost):
        self.model = model
        self.cost=cost
    def __call__(self, x):
        X = x.reshape(1,-1)
        cu_fid=int(X[:, -1] )
        X[:, -1] =cu_fid
        py, var = self.model.predict(X)
        sigma = np.sqrt(abs(var))
        self.sigma = sigma
        py=np.log(1+np.exp(py))/self.cost[cu_fid]
        self.AC_value =py
        self.py =py
        return py[0]


#
# from scipy.stats import norm
# from pymoo.core.problem import Problem
# import numpy as np
#
#
# class AcquisitionMB1(Problem):
#     def __init__(self, n_var, model,modelc, n_obj, xl, xu,):
#         super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, type_var=np.double)
#         self.model = model
#         self.modelc=modelc
#     def _evaluate(self, x, out, *args, **kwargs):
#         X =x.copy()
#
#         py, var = self.model.predict(X)
#         sigma = np.sqrt(abs(var))
#         pre_cost,v = self.modelc.predict(X)
#         y = self.model.Y
#         fmin = min(y)
#         Z = (fmin - py) / sigma
#         ei = (fmin - py) * norm.cdf(Z) + sigma * norm.pdf(Z)
#         self.ei = ei
#         self.AC_value = -ei
#         self.sigma = sigma
#         out["F"] = np.column_stack([-self.ei[:,0],pre_cost[:,0]])
#
#     def _compute_acq(self, x):
#         X = x.copy()
#
#         py, var = self.model.predict(X)
#         sigma = np.sqrt(abs(var))
#         pre_cost,v = self.modelc.predict(X)
#         y = self.model.Y
#         fmin = min(y)
#         Z = (fmin - py) / sigma
#         ei = (fmin - py) * norm.cdf(Z) + sigma * norm.pdf(Z)
#         self.ei = ei
#         self.AC_value = -ei
#         self.sigma = sigma
#         # return -self.ei[0],-pre_cost[0]
#         return np.column_stack([-self.ei[:,0],pre_cost[:,0]])
#
#
# class AcquisitionMB(Problem):
#     def __init__(self, n_var, model,modelc, n_obj, xl, xu,target):
#         super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, type_var=np.double)
#         self.model = model
#         self.modelc=modelc
#         self.target=target
#
#     def _evaluate(self, x, out, *args, **kwargs):
#         X=np.zeros((x.shape[0],self.n_var+1))
#         X[:,:-1] =x
#         X[:,-1]=self.target
#
#         py, var = self.model.predict(X)
#         sigma = np.sqrt(abs(var))
#         # pre_cost,v = self.modelc.predict(X)
#         # y = self.model.Y
#         # fmin = min(y)
#         # Z = (fmin - py) / sigma
#         # ei = (fmin - py) * norm.cdf(Z) + sigma * norm.pdf(Z)
#         # self.ei = ei
#         # self.AC_value = -ei
#         self.sigma = sigma
#         out["F"] = np.column_stack([py[:,0],-sigma[:,0]])
#
#     def _compute_acq(self, x):
#         X = np.zeros((x.shape[0], self.n_var + 1))
#         X[:, :-1] = x
#         X[:,-1]=self.target
#
#         py, var = self.model.predict(X)
#         sigma = np.sqrt(abs(var))
#         # pre_cost,v = self.modelc.predict(X)
#         # y = self.model.Y
#         # fmin = min(y)
#         # Z = (fmin - py) / sigma
#         # ei = (fmin - py) * norm.cdf(Z) + sigma * norm.pdf(Z)
#         # self.ei = ei
#         # self.AC_value = -ei
#         self.sigma = sigma
#         # return -self.ei[0],-pre_cost[0]
#         return np.column_stack([py[:,0],-sigma[:,0]])
