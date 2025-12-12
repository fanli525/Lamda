
import numpy as np
from task_similarity import IoUTaskSimilarity

def cal_cov(X,F,cs,ns,ni=20):
    name = cs.get_hyperparameter_names()
    X1=[]
    for i in range(len(name)):
        config = cs.get_hyperparameter(name[i])
        config_type = config.__class__.__name__
        is_ordinal = config_type.startswith("Ordinal")
        is_categorical = config_type.startswith("Categorical")
        if is_categorical ==True:
            X1.append(X[:,i])
        else:
            X1.append(X[:,i].astype(np.float32))

    samp, re_all = [], []
    j=X.shape[0] - ns

    observations_set = []
    for j1 in [j, j + ns]:
        ob = {f"{name[d]}": X1[d][:j1] for d in range(X.shape[1])}
        ob["loss"] = F[:j1]
        observations_set.append(ob)

    ts = IoUTaskSimilarity(n_samples=60000, config_space=cs, observations_set=observations_set,
                           promising_quantile=0.3,dim_reduction_factor = 1)
    re = ts.compute(method="total_variation")[0, 1]# we change here, ovl=1-total_variation
    # print( ' sim: ', re)
    samp.append(j), re_all.append(re)
    return re_all





def cal_cov_opt(X,F,cs,ns,ni=20):
    name = cs.get_hyperparameter_names()
    samp, re_all = [], []
    for j in range(X.shape[1] + ns, F.shape[0] - ns):

        observations_set = []
        for j1 in [j, -1]:
            ob = {f"{name[d]}": X[:j1, d] for d in range(X.shape[1])}
            ob["loss"] = F[:j1]
            observations_set.append(ob)

        ts = IoUTaskSimilarity(n_samples=60000, config_space=cs, observations_set=observations_set,
                               promising_quantile=0.3,dim_reduction_factor = 1)
        re = ts.compute(method="total_variation")[0, 1]
        print( ' sim: ', re)
        samp.append(j), re_all.append(re)
    return re_all





def cal_cov_realopt(X,F,cs,X_real,F_real):
    name = cs.get_hyperparameter_names()
    samp, re_all = [], []
    for j in range(X.shape[1] , F.shape[0]):

        observations_set = []
        for j1 in [j, -1]:
            if j1==-1:
                ob = {f"{name[d]}": X_real[:, d] for d in range(X.shape[1])}
                ob["loss"] = F_real
                observations_set.append(ob)

            else:
                ob = {f"{name[d]}": X[:j1, d] for d in range(X.shape[1])}
                ob["loss"] = F[:j1]
                observations_set.append(ob)

        ts = IoUTaskSimilarity(n_samples=60000, config_space=cs, observations_set=observations_set,
                               promising_quantile=0.3,dim_reduction_factor = 1)
        re = ts.compute(method="total_variation")[0, 1]
        print( ' sim: ', re)
        samp.append(j), re_all.append(re)
    return re_all

