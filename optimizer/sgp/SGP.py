
from scipy.spatial.distance import cdist
from optimizer.utilit.init_latin_hypercube_sampling import  init_latin_hypercube_sampling
from optimizer.utilit.Initialization_mix import initial_model
import time,random,copy
from optimizer.sgp.Acq import  MF_sLCB, MF_sEI
from scipy.optimize import differential_evolution
import numpy as np
from utilit.fit_pdf import FitPDF
from utilit.stop_cond import stop_vol
class MF_Opt: 
    def __init__(self,fo,Param,cs,plot = False,kde_vartypes = 'cc',vartypes = []):
        self.fo=fo
        self.Param = Param
        self.plot = False
        self.cs=cs
        self.kde_vartypes = kde_vartypes
        self.vartypes = vartypes
        self.stop_vol=stop_vol(cs=cs)
    def __call__(self):
        Param=copy.deepcopy(self.Param)
        np.random.seed(Param.r)
        fo=self.fo
        terget_fid=fo.x_fid[-1]
        problem_bounds=copy.deepcopy(np.array(fo.bound))
        if len(fo.d_log)>0:
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

        hy_bd=list(zip(list(Tbd),list(Tbu)))
        Nt=int(sum(N_train))
        xint=init_latin_hypercube_sampling(np.array(Tbd), np.array(Tbu), Nt,np.random.RandomState(Param.r))
        per_t=int(Nt/len(fo.x_fid))
        for i in range(len(fo.x_fid)):
            xint[i*per_t:(i+1)*per_t,-1]=fo.x_fid[i]
        if len(Param.prior)>0:
            X_prior=Param.prior[0]['Samp']
            Y_prior=Param.prior[0]['YS']
            C_prior=Param.prior[0]['cost_all']
            Param.bd_prom = Param.prior[0]['bd_prom']
        else:
            X_prior=[]
            Y_prior=[]
            C_prior=[]
            Param.bd_prom =[]
        if len(X_prior)>0:
            Xt = np.array(X_prior)
            Xt1 =np.array(X_prior)
        else:
            Xt= xint.copy()
            Xt1= xint.copy()
        Xt[:,-1]= 0
        if len(Param.prior)==0:
            if len(fo.d_log)>0:
                Xt1[:, fo.d_log] = np.exp(Xt1[:, fo.d_log])
            if len(fo.d_int)>0:
                Xt1[:, fo.d_int] = np.round(Xt1[:, fo.d_int])
        else:
            if len(fo.d_log) > 0:
                Xt[:, fo.d_log] = np.log(Xt[:, fo.d_log])
        Yt, cost = [], []
        if len(X_prior)==0:
            for i in range(Xt1.shape[0]):
                y, c = fo.fit(Xt1[i, :-1], Xt1[i, -1])
                Yt.append(y), cost.append(c)
            Yt = np.array(Yt)
            cost = np.array(cost)
        else:
            for i in range(Xt1.shape[0]):
                y, c = Y_prior[i], C_prior[i]
                Yt.append(y), cost.append(c)
            Yt = np.array(Yt)
            cost = np.array(cost)

        Yt = Yt[:, np.newaxis]
        Yt_ = Yt.reshape(-1, fo.task, order='F')
        cost_ = cost.reshape(-1, fo.task, order='F')
        cost_all = cost
        YC=cost_all
        YC= YC[:, np.newaxis]
        Samp = copy.deepcopy(Xt1)
        YS = copy.deepcopy(Yt).reshape(-1,1)
        maxcost =Param.maxcost
        # fo.cost=[1,3]
        R2 ,RS,Ymin,Cost= [],[],[],[]
        rt = 0
        inc_valid = np.inf
        runtime = []
        for i in range(len(YS)):
            if Samp[i,-1] == terget_fid:
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
            model=initial_model(Xt,YS,fo,model_M,normalization,d_int=False,Kint=Param.Kint)
        m,X_train,Y_train=model()
        it,npv,stop_cond =0,0,False
        while  runtime[-1]<Param.maxcost and stop_cond==False:
            print(model_M,"runtime/maxcos: ", runtime[-1], '/', maxcost, 'y_opt: ', np.min(Ymin))
            print(model_M,"it: ", it)

            if norm_Y == 'log':
                m.set_data(Xt, np.log(YS))
            else:
                m.set_data(Xt, YS)
            m.optimize()
            next_task=terget_fid
            fun = MF_sEI(Param.FM_name, m, next_task)
            if len(Param.prior) > 0:
                weighted_samples = pdf_fitter.sample_weighted(data_min=np.array(Tbd[:-1]), data_max=np.array(Tbu[:-1]),
                                                              n_samples=1000, alpha=0.5)
                candi = np.array(weighted_samples)
                ac_val = fun(candi)
                ind=np.argmin(ac_val)
                next_x = candi[ind, :]
            else:
                bound_opt = hy_bd[:-1].copy()
                optparams = differential_evolution(fun, bounds=bound_opt, maxiter=500)
                next_x = optparams.x
            final_x = np.zeros((1, fo.d+ 1))
            final_x[0, :-1] = next_x.reshape(1,-1)
            final_x[0, -1] = fo.x_fid[-1]
            final_x1=final_x.copy()
            if len(fo.d_log)>0:
                final_x1[:,fo.d_log]=np.exp(final_x1[:,fo.d_log])
            X_new=final_x.reshape(-1,fo.d+1)
            npv=0
            for i in range(X_new.shape[0]):
                X_new[i, :-1][(X_new[i, :-1]>problem_bounds[:,1])]=problem_bounds[:,1][(X_new[i, :-1]>problem_bounds[:,1])]
                X_new[i, :-1][(X_new[i, :-1]<problem_bounds[:,0])]=problem_bounds[:,0][(X_new[i, :-1]<problem_bounds[:,0])]

                x_new =  copy.copy(X_new[i,:].reshape(1,-1))
                x_new1 =x_new.copy()
                x_new1[:,-1]=0

                dist = cdist(x_new, Xt, metric='euclidean')
                if np.min(dist) > dalta :
                    if len(fo.d_log)>0:
                        x_new[:,fo.d_log] = np.exp(x_new[:,fo.d_log])
                    if len(fo.d_int)>0:
                        x_new[:,fo.d_int] = np.round(x_new[:,fo.d_int])

                    # x_new[:,-1] =  np.round(np.exp(x_new[:,-1]))
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

                    if x_new[0][-1] == terget_fid:
                        if inc_valid > y_new:
                            inc_valid = y_new
                            npv =1
                    Ymin.append(float(inc_valid))
                    rt += cost_all[-1]
                    runtime.append(float(rt))
                    fo.runtime = runtime
                    Param.Ymin = Ymin
            it = it + 1
            if  len(Param.prior) == 0:
                stop_cond=self.stop_vol(Samp[:,:-1],YS)

        T2 = time.time()
        y_opt = min(YS[Samp[:, -1] == terget_fid])
        Ib = np.where(YS == y_opt)[0][0]
        x_opt = Samp[Ib, :-1]
        ind = np.argsort(YS[:,0])[::1]
        n = min(int(len(YS[:,0]) / 3), 10 * (x_opt.shape[0]))
        X_prom = Samp[ind[:n], :]
        min_values = np.min(X_prom, axis=0)
        max_values = np.max(X_prom, axis=0)
        bd_prom = np.vstack((min_values, max_values)).T
        EV_inf.append({
            "xopt": x_opt.tolist(), "yopt": y_opt.tolist(),"X_prom": X_prom.tolist(),
            "Samp": Samp.tolist(), "YS": YS.tolist(),
            "cost_all": cost_all.tolist(), "runtime": runtime,
            "Ymin": Ymin, 'Time': T2-T1,
            'pre_inf_all':pre_inf_all,  'R2': R2,'bd_prom': bd_prom
        })
        return EV_inf


