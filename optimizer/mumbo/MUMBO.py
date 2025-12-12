import time,random
import numpy as np
from optimizer.utilit.init_latin_hypercube_sampling import  init_latin_hypercube_sampling
from optimizer.utilit.Initialization_mix import initial_model,initial_model_cost
from scipy.spatial.distance import cdist
from emukit.bayesian_optimization.acquisitions.max_value_entropy_search import MUMBO
from emukit.core.acquisition import Acquisition
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.loop.candidate_point_calculators import GreedyBatchPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
import copy,os
from utilit.fit_pdf import FitPDF
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from utilit.stop_cond import stop_vol


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
    def __init__(self,fo,Param,plot = False,cs='',kde_vartypes = 'cc',vartypes = []):
        self.fo=fo
        self.Param = Param
        self.plot = False
        self.kde_vartypes = kde_vartypes
        self.vartypes = vartypes
        self.cs=cs
        self.stop_vol=stop_vol(cs=cs)

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
        if len(Param.prior)>0:
            train_data_good = np.array(Param.prior[0]['X_prom'])[:,:-1]  # 2D 数据
            if len(fo.d_log) > 0:
                train_data_good[:, fo.d_log] = np.exp(train_data_good[:, fo.d_log])
            if len(fo.d_int) > 0:
                train_data_good[:, fo.d_int] = np.round(train_data_good[:, fo.d_int])
            pdf_fitter = FitPDF(self.cs, train_data_good, self.kde_vartypes)
            pdf_fitter.fit_pdf_pro()

        Nt=int(sum(N_train))
        xint=init_latin_hypercube_sampling(np.array(Tbd), np.array(Tbu), Nt,np.random.RandomState(Param.r))
        per_t=int(Nt/len(fo.x_fid))
        xint[:, -1] = fo.x_fid[-1]
        if len(Param.prior)==0:
            for i in range(len(fo.x_fid)):
                xint[i*per_t:(i+1)*per_t,-1]=fo.x_fid[i]
        else:
            xint[:,-1]=fo.x_fid[-1]

        if len(Param.prior)>0:
            X_prior=np.array(Param.prior[0]['Samp'])
            Y_prior=Param.prior[0]['YS']
            C_prior=Param.prior[0]['cost_all']
            Param.bd_prom = Param.prior[0]['bd_prom']
            xint=np.vstack((X_prior,xint ))
        Xt= xint.copy()
        Xt1= xint.copy()
        Xt[:, -1] = 1
        if len(Param.prior)==0:
            for i in range(len(fo.x_fid)):
                Xt[i*per_t:(i+1)*per_t,-1]=i
        else:
            Xt[:-Nt,-1]=0
            Xt[-Nt:,-1]=1
        if len(fo.d_log)>0:
            Xt1[:, fo.d_log] = np.exp(Xt1[:, fo.d_log])
        if len(fo.d_int)>0:
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
        YS = copy.deepcopy(Yt).reshape(-1,1)
        cost_HF = np.mean(cost[Xt1[:,-1]==fo.x_fid[-1]])

        maxcost =Param.maxcost
        Ymin,rt  = [],Param.rt
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
        if norm_Y=='log':
            model=initial_model(Xt,np.log(YS),fo,model_M,normalization,d_int=False,Kint=Param.Kint)
        else:
            model=initial_model(Xt,YS,fo,model_M,normalization,d_int=False,Kint=Param.Kint)
        modelc=initial_model_cost(Xt,YC,fo,model_M,normalization,d_int=False)
        m,X_train,Y_train=model()
        mc=modelc()
        it,npv,p0 =0,0,0.8
        batch_size=3
        if len(Param.prior)>0:
            bd_prom = Param.bd_prom.copy()
        else:
            Param.bd_prom,bd_prom=[],[]
        if len(Param.bd_prom) > 0:
            if len(fo.d_log) > 0:
                bd_prom[fo.d_log, :] = np.log(bd_prom[fo.d_log, :])

        while runtime[-1]<maxcost:
            print(model_M,"runtime/maxcos: ", runtime[-1], '/', maxcost, 'y_opt: ', np.min(Ymin))
            mc.set_data(Xt, YC)
            mc.optimize()
            if norm_Y == 'log':
                m.set_data(Xt, np.log(YS))
            else:
                m.set_data(Xt, YS)
            m.optimize()
            p1 = p0**(it)

            if len(bd_prom)>0 and   random.random()<p1:
                space = [ContinuousParameter('x' + str(i), bd_prom[:,0][i], bd_prom[:,1][i]) for i in range(fo.d)]
            else:
                space = [ContinuousParameter('x' + str(i), Tbd[i], Tbu[i]) for i in range(fo.d)]
            s2 = space.append(InformationSourceParameter(fo.task))
            s2 = space
            parameter_space = ParameterSpace(s2)

            cost_acquisition = Cost_fun(mc)
            mumbo_acquisition = MUMBO(m, parameter_space, num_samples=10, grid_size=500 * fo.d) / cost_acquisition
            acquisition = mumbo_acquisition
            initial_loop_state = create_loop_state(Xt, np.log(YS))
            acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(parameter_space, num_anchor=3),
                                                                    parameter_space)
            if len(Param.prior) > 0:
                weighted_samples = pdf_fitter.sample_weighted(data_min=np.array(Tbd[:-1]), data_max=np.array(Tbu[:-1]),
                                                              n_samples=1000, alpha=0.5)
                candi = np.array(weighted_samples)
                xtest_m=[]
                for i in range(fo.task):
                    xtest_m.append(candi)
                Xtest1 = convert_x_list_to_array(xtest_m)
                ac_val = mumbo_acquisition.evaluate(Xtest1)
                ind = np.argmax(ac_val)  # 最大值的索引
                new_x = Xtest1[ind, :]
            else:
                candidate_point_calculator = GreedyBatchPointCalculator(m, acquisition, acquisition_optimizer, batch_size)
                try:
                    new_x = candidate_point_calculator.compute_next_points(initial_loop_state)
                except:
                    new_x=[]
            if len(new_x)>0:
                X_new = np.atleast_2d(new_x)
            else:
                X_new = np.atleast_2d(init_latin_hypercube_sampling(np.array(Tbd), np.array(Tbu), 1, np.random.RandomState(Param.r)))
                X_new[-1]=int( random.choice(np.linspace(0,len(fo.x_fid)-1,len(fo.x_fid))))
            N_new = 0
            for i in range(X_new.shape[0]):
                X_new[i, :-1][(X_new[i, :-1]>problem_bounds[:,1])]=problem_bounds[:,1][(X_new[i, :-1]>problem_bounds[:,1])]
                X_new[i, :-1][(X_new[i, :-1]<problem_bounds[:,0])]=problem_bounds[:,0][(X_new[i, :-1]<problem_bounds[:,0])]

                x_new =  copy.copy(X_new[i,:].reshape(1,-1))
                x_new1 =x_new.copy()
                x_new[:,-1]=fo.x_fid[int(x_new[:, -1])]
                dist = cdist(x_new, Xt1, metric='euclidean')
                if np.min(dist) > dalta :
                    if len(fo.d_log) > 0:
                        x_new[:,fo.d_log] = np.exp(x_new[:,fo.d_log])
                    if len(fo.d_int) > 0:
                        x_new[:,fo.d_int] = np.round(x_new[:,fo.d_int])
                    x_new[0, :-1][(x_new[0, :-1] > np.array(fo.bound)[:, 1])] = np.array(fo.bound)[:, 1][(x_new[0, :-1] > np.array(fo.bound)[:, 1])]
                    x_new[0, :-1][(x_new[0, :-1] < np.array(fo.bound)[:, 0])] = np.array(fo.bound)[:, 0][(x_new[0, :-1] < np.array(fo.bound)[:, 0])]
                    y_new,c_new = fo.fit(x_new[0,:-1],x_new[0,-1])
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
        y_opt = min(YS[Samp[:, -1] == fo.x_fid[-1]])
        Ib = np.where(YS == y_opt)[0][0]
        x_opt = Samp[Ib, :-1]

        EV_inf.append({
            "xopt": x_opt.tolist(), "yopt": y_opt.tolist(),
            "Samp": Samp.tolist(), "YS": YS.tolist(),
            "cost_all": cost_all.tolist(), "runtime": runtime,
            "Ymin": Ymin, 'Time': T2-T1,
        })
        return EV_inf


