import time, random
import numpy as np
from optimizer.utilit.init_latin_hypercube_sampling import init_latin_hypercube_sampling
from optimizer.utilit.Initialization_mix import initial_model, initial_model_cost
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import copy, os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class MF_Opt:
    def __init__(self, fo, Param, plot=False):
        self.fo = fo
        self.Param = Param
        self.plot = False

    def __call__(self):
        Param = copy.deepcopy(self.Param)
        np.random.seed(Param.r)
        fo = self.fo
        dimension = copy.deepcopy(fo.d)
        problem_bounds = copy.deepcopy(np.array(fo.bound))
        problem_bounds[fo.d_log, :] = np.log(problem_bounds[fo.d_log, :])
        model_M = copy.deepcopy(Param.model_M)
        N_train = copy.deepcopy(Param.N_train)
        Tbd = problem_bounds[:, 0].tolist()
        Tbd.append(fo.bound_fid[0])
        Tbu = problem_bounds[:, 1].tolist()
        Tbu.append(fo.bound_fid[1])
        Nt = int(sum(N_train))

        xint2,xint3,xint4=[],[],[]
        for i in range(len(fo.x_fid)):
            xint = init_latin_hypercube_sampling(np.array(Tbd), np.array(Tbu), int(Nt/(i+1)), np.random.RandomState(Param.r))
            xint3_1 = copy.deepcopy(xint)
            xint3_1[:, -1] = i
            xint4_1 = copy.deepcopy(xint)
            xint4_1[:, -1] = fo.x_fid[i]
            if i==0:
                xint2=xint
                xint3=xint3_1
                xint4=xint4_1
            else:
                xint2=np.vstack((xint2,xint))
                xint3=np.vstack((xint3,xint3_1))
                xint4=np.vstack((xint4,xint4_1))

        Xt = xint3.copy()
        Xt1 = xint4.copy()
        xint=xint2
        ntest=1000
        xint1 = init_latin_hypercube_sampling(np.array(Tbd), np.array(Tbu), ntest, np.random.RandomState(Param.r))
        xint1 = np.tile(xint1, (len(fo.x_fid), 1))
        xtest=copy.deepcopy(xint1)
        for i in range(len(fo.x_fid)):
            xtest[i*ntest:(i+1)*ntest,-1]=fo.x_fid[i]
            xint1[i*ntest:(i+1)*ntest,-1]=i
        Ytest, cost_test = [], []
        for i in range(xtest.shape[0]):
            y, c = fo.fit(xtest[i, :-1], xtest[i, -1])
            Ytest.append(y), cost_test.append(c)
        Ytest=np.array(Ytest)
        per_t = int(np.ceil(Nt / len(fo.x_fid)))

        # for i in range(len(fo.x_fid)):
        #     xint[i * per_t:(i + 1) * per_t, -1] = fo.x_fid[i]
        # Xt = xint.copy()
        # Xt1 = xint.copy()
        #
        # for i in range(len(fo.x_fid)):
        #     Xt[i * per_t:(i + 1) * per_t, -1] = i

        if len(fo.d_log) > 0:
            Xt1[:, fo.d_log] = np.exp(Xt1[:, fo.d_log])
        if len(fo.d_int) > 0:
            Xt1[:, fo.d_int] = np.round(Xt1[:, fo.d_int])
        Yt, cost = [], []
        for i in range(Xt1.shape[0]):
            y, c = fo.fit(Xt1[i, :-1], Xt1[i, -1])
            Yt.append(y), cost.append(c)
        Yt = np.array(Yt)
        cost = np.array(cost)
        Yt = Yt[:, np.newaxis]
        cost_all = cost
        YC = cost_all
        YC = YC[:, np.newaxis]
        Samp = copy.deepcopy(Xt1)
        YS = copy.deepcopy(Yt).reshape(-1, 1)


        Ymin, rt = [], Param.rt
        inc_valid = np.inf
        runtime = []
        for i in range(len(YS)):
            if Samp[i, -1] == fo.bound_fid[1]:
                if inc_valid > YS[i, 0]:
                    inc_valid = YS[i, 0]
            Ymin.append(float(inc_valid))
            rt += cost_all[i]
            runtime.append(float(rt))

        Param.Ymin = Ymin
        fo.runtime = runtime
        dalta = 1e-6
        it, N_new, normalization = 1, 0, True
        T1 = time.time()
        EV_inf, pre_inf_all = [], []
        norm_Y = Param.NorY
        if norm_Y == 'log':
            model = initial_model(Xt, np.log(YS), fo, model_M, normalization, d_int=False, Kint=Param.Kint)
        else:
            model = initial_model(Xt, YS, fo, model_M, normalization, d_int=False, Kint=Param.Kint)

        m, X_train, Y_train = model()
        try:
            m.optimize()
        except:
            pass

        mse_all,r2_all,mean, sigma=[],[],np.zeros_like(Ytest),[]
        try:
            mean, sigma = m.predict(xint1)
            for i in range(len(fo.x_fid)):
                mse = np.mean((mean[xint1[:,-1]==i, 0] - Ytest[xint1[:,-1]==i]) ** 2)
                r2= r2_score(mean[xint1[:,-1]==i, 0] , Ytest[xint1[:,-1]==i])
                mse_all.append(mse)
                r2_all.append(r2)
        except:
            pass

        T2 = time.time()


        EV_inf.append({
            "mse_all": mse_all, "r2_all":r2_all,
            "Samp": Samp.tolist(), "YS": YS.tolist(),
            "mean": mean.tolist(), "Ytest": Ytest,
             'Time': T2 - T1,
        })
        return EV_inf
