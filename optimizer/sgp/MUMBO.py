import time
import numpy as np
from optimizer.utilit.init_latin_hypercube_sampling import  init_latin_hypercube_sampling
from optimizer.utilit.Initialization_mix import initial_model,initial_model_cost
from scipy.spatial.distance import cdist
from optimizer.emukit.bayesian_optimization.acquisitions.max_value_entropy_search import MUMBO
from optimizer.emukit.core.acquisition import Acquisition
from optimizer.emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from optimizer.emukit.core.loop import GreedyBatchPointCalculator
from optimizer.emukit.core.loop.loop_state import create_loop_state
from optimizer.emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from optimizer.emukit.core.optimization import GradientAcquisitionOptimizer
import copy,os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Cost_fun(Acquisition):
    def __init__(self, mc):
        self.mc = mc

    def evaluate(self, x):
        mu,var=self.mc.predict(x)

        return mu

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evalute(x), np.zeros(x.shape)








class MF_Opt: 
    def __init__(self,fo,Param,plot = False):
        self.fo=fo
        self.Param = Param
        self.plot = False

    def __call__(self):
        Param=copy.deepcopy(self.Param)
        np.random.seed(Param.r)
        fo=self.fo
        dimension=copy.deepcopy(fo.d)
        problem_bounds=copy.deepcopy(np.array(fo.bound))
        problem_bounds[fo.d_log,:]=np.log(problem_bounds[fo.d_log,:])
        model_M=copy.deepcopy(Param.model_M)
        N_train=copy.deepcopy(Param.N_train)
        Tbd= problem_bounds[:,0].tolist()
        Tbd.append(fo.bound_fid[0])
        Tbu=problem_bounds[:,1].tolist()
        Tbu.append(fo.bound_fid[1])
        Nt=int(sum(N_train))
        xint=init_latin_hypercube_sampling(np.array(Tbd), np.array(Tbu), Nt,np.random.RandomState(Param.r))
        per_t=int(Nt/len(fo.x_fid))
        for i in range(len(fo.x_fid)):
            xint[i*per_t:(i+1)*per_t,-1]=fo.x_fid[i]

        Xt= xint.copy()
        Xt1= xint.copy()
        for i in range(len(fo.x_fid)):
            Xt[i*per_t:(i+1)*per_t,-1]=i
        Xt1[:, fo.d_log] = np.exp(Xt1[:, fo.d_log])
        Xt1[:, fo.d_int] = np.round(Xt1[:, fo.d_int])
        Yt,cost = [],[]
        for i in range(Xt1.shape[0]):
            y,c=fo.fit(Xt1[i,:-1],Xt1[i,-1])
            Yt.append(y), cost.append(c)
        Yt = np.array(Yt)
        cost = np.array(cost)
        Yt = Yt[:, np.newaxis]
        cost_all = cost
        YC=cost_all
        YC= YC[:, np.newaxis]
        Samp = copy.deepcopy(Xt1)
        YS = copy.deepcopy(Yt)
        cost_HF = np.mean(cost[Xt1[:,-1]==fo.x_fid[-1]])
        maxcost = max(cost_HF * 30, 10000)
        Ymin,rt  = [],0
        inc_valid = np.inf
        runtime = []
        for i in range(len(YS)):
            if Samp[i,-1] == fo.bound_fid[1]:
                if inc_valid > YS[i,0]:
                    inc_valid = YS[i,0]
            Ymin.append(float(inc_valid))
            rt += cost_all[i]
            runtime.append(float(rt))

        Param.Ymin = Ymin
        fo.runtime = runtime
        dalta=1e-6
        it,N_new,normalization=1, 0,True
        T1=time.time()
        EV_inf,pre_inf_all=[],[]
        norm_Y=Param.NorY
        if len(fo.d_int)>0:
            d_int=True
        else:
            d_int=False


        if norm_Y=='log':
            model=initial_model(Xt,np.log(YS),fo,model_M,normalization,d_int=d_int,Kint=Param.Kint)
        else:
            model=initial_model(Xt,YS,fo,model_M,normalization,d_int=d_int,Kint=Param.Kint)
        modelc=initial_model_cost(Xt,YC,fo,model_M,normalization,d_int=False)
        # model=initial_model(Xt,YS,fo,model_M,normalization,d_int=fo.d_int)
        m,X_train,Y_train=model()
        mc=modelc()

        it,npv =0,0
        batch_size=2
        while runtime[-1]<maxcost:
            print(model_M,"runtime/maxcos: ", runtime[-1], '/', maxcost, 'y_opt: ', np.min(Ymin))
            mc.set_data(Xt, YC)
            mc.optimize()
            if norm_Y == 'log':
                m.set_data(Xt, np.log(YS))
            else:
                m.set_data(Xt, YS)
            m.optimize()

            space = [ContinuousParameter('x' + str(i), Tbd[i], Tbu[i]) for i in range(fo.d)]
            # s2 = space.append(InformationSourceParameter(Tbu[-1]))
            # s2 = space.append(InformationSourceParameter(fo.x_fid))
            s2 = space.append(InformationSourceParameter(fo.task))

            s2 = space
            parameter_space = ParameterSpace(s2)

            cost_acquisition = Cost_fun(mc)
            mumbo_acquisition = MUMBO(m, parameter_space, num_samples=10, grid_size=500 * fo.d) / cost_acquisition

            acquisition = mumbo_acquisition
            initial_loop_state = create_loop_state(Xt, np.log(YS))
            acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(parameter_space, num_anchor=3),
                                                                    parameter_space)
            candidate_point_calculator = GreedyBatchPointCalculator(m, acquisition, acquisition_optimizer, batch_size)
            new_x = candidate_point_calculator.compute_next_points(initial_loop_state)
            X_new = np.atleast_2d(new_x)
            N_new = 0
            for i in range(X_new.shape[0]):
                X_new[i, :-1][(X_new[i, :-1]>problem_bounds[:,1])]=problem_bounds[:,1][(X_new[i, :-1]>problem_bounds[:,1])]
                X_new[i, :-1][(X_new[i, :-1]<problem_bounds[:,0])]=problem_bounds[:,0][(X_new[i, :-1]<problem_bounds[:,0])]

                x_new =  copy.copy(X_new[i,:].reshape(1,-1))
                x_new1 =x_new.copy()
                x_new[:,-1]=fo.x_fid[int(x_new[:, -1])]
                dist = cdist(x_new, Xt1, metric='euclidean')
                if np.min(dist) > dalta :
                    x_new[:,fo.d_log] = np.exp(x_new[:,fo.d_log])
                    x_new[:,fo.d_int] = np.round(x_new[:,fo.d_int])

                    # x_new[:,-1] =  np.round(np.exp(x_new[:,-1]))
                    x_new[0, :-1][(x_new[0, :-1] > np.array(fo.bound)[:, 1])] = np.array(fo.bound)[:, 1][(x_new[0, :-1] > np.array(fo.bound)[:, 1])]
                    x_new[0, :-1][(x_new[0, :-1] < np.array(fo.bound)[:, 0])] = np.array(fo.bound)[:, 0][(x_new[0, :-1] < np.array(fo.bound)[:, 0])]
                    y_new,c_new = fo.fit(x_new[:,:-1],x_new[:,-1])


                    x_fid = int(x_new[:, -1])
                    print( 'x_fid: ', x_fid,' y_new:', y_new)
                    Xt = np.vstack((Xt, x_new1))
                    Samp = np.vstack((Samp, x_new))
                    YS = np.append(YS, y_new)
                    YC = np.append(YC, c_new)

                    cost_all = np.append(cost_all, c_new)
                    YS = YS[:, np.newaxis]
                    YC = YC[:, np.newaxis]

                    if x_new[0][-1] == fo.bound_fid[1]:
                        if inc_valid > y_new:
                            inc_valid = y_new
                    Ymin.append(float(inc_valid))
                    rt += cost_all[-1]
                    runtime.append(float(rt))
                    fo.runtime = runtime
                    Param.Ymin = Ymin

                else:
                    N_new = N_new+1


            it = it + 1


        T2 = time.time()
        y_opt = min(YS[Samp[:, -1] == Tbu[-1]])
        Ib = np.where(YS == y_opt)[0][0]
        x_opt = Samp[Ib, :-1]

        EV_inf.append({
            "xopt": x_opt.tolist(), "yopt": y_opt.tolist(),
            "Samp": Samp.tolist(), "YS": YS.tolist(),
            "cost_all": cost_all.tolist(), "runtime": runtime,
            "Ymin": Ymin, 'Time': T2-T1,
        })
        return EV_inf


