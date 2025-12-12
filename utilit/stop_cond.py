from ConfigSpace import ConfigurationSpace
import numpy as np
from task_similarity.cal_sim import cal_cov
class stop_vol:
    def __init__(self, cs,  gmma0=0.1,ns=5):
        self.cs = cs
        self.gmma0 = gmma0
        self.ns = ns
    def __call__(self,X,F):
        X = np.array(X)
        cs = self.cs
        name = cs.get_hyperparameter_names()
        ind = []
        X1 = []
        cs1 = ConfigurationSpace()
        for dim in range(len(cs)):
            config_type = cs[name[dim]].__class__.__name__
            is_int = config_type.startswith("Categorical")
            if is_int == False:
                ind.append(dim)
                xa = X[:, dim].astype(float)
                X1.append(xa)
                cs1.add_hyperparameter(cs[name[dim]])
        X2 = np.array(X1).T
        sim = cal_cov(X2, np.array(F[:,0]), cs1, ns=self.ns)
        if  np.abs(1 - sim[-1]) < self.gmma0:  # n_iterations<0 or
            stop=True
        else:
            stop=False
        return stop