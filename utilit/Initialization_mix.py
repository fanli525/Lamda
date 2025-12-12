import numpy as np
# from  optimizer.mumbo import GPy
import GPy
import optimizer.mumbo.emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper,GPyModelWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from sklearn.linear_model import LinearRegression

class initial:
    def __init__(self,N_train,maxN=None,FM_name=None,AC=None,model_M=None,lh_threshold=None,c_threshold=None,r=None):
        self.N_train=N_train
        self.maxN=maxN
        self.AC=AC
        self.FM_name=FM_name
        self.model_M=model_M
        self.lh_threshold=lh_threshold
        self.c_threshold=c_threshold
        self.r=r
        self.Ymin=None

    def __call__(self):

        return self


class initial_model:
    def __init__(self,Xt,YS,fo,model_M,normalization=False,d_int=False,ARD=False,Kint='Linear',Ker_type='single'):
        self.Xt=Xt
        self.YS=YS
        self.fo=fo
        self.model_M=model_M
        self.normalization=normalization
        self.d_int=d_int
        self.ARD = ARD
        self.Kint = Kint
        self.Ker_type = Ker_type

    def __call__(self):
        Xt=self.Xt
        YS=self.YS
        fo=self.fo
        model_M = self.model_M
        N_Restar=1
        X_train = []
        Y_train = []
        for i in range(fo.task):
            X_train.append(Xt[Xt[:, -1] == i, :-1])
            Y_train.append(YS[Xt[:, -1] == i, :])


        if self.d_int==False:
            if model_M == 'SGP':
                K = GPy.kern.Matern52(fo.d,ARD=self.ARD)+ GPy.kern.Bias(fo.d)
                m = GPy.models.GPRegression(X_train[-1], Y_train[-1], kernel=K,normalizer=self.normalization)
                m.Gaussian_noise.fix(1.0e-8)
                m = GPyModelWrapper(m, n_restarts=N_Restar)

            elif model_M == 'ICM':
                K = GPy.kern.Matern52(fo.d,ARD=self.ARD)+ GPy.kern.Bias(fo.d)
                icm = GPy.util.multioutput.ICM(input_dim=fo.d, num_outputs=fo.task, kernel=K, W_rank=fo.task)
                m = GPy.models.GPCoregionalizedRegression(X_train, Y_train, kernel=icm,normalizer=self.normalization)
                m['.*noise*'].constrain_fixed(1.0e-8)
                m = GPyMultiOutputWrapper(m, fo.task, n_optimization_restarts=N_Restar)
            elif model_M == 'NAR':
                base_kernel = GPy.kern.Matern52
                kernels = make_non_linear_kernels(base_kernel, fo.task, fo.d, 'single')
                m = NonLinearMultiFidelityModel(Xt, YS, n_fidelities=fo.task, kernels=kernels,verbose=True, optimization_restarts=N_Restar,normalizer=self.normalization)
                for m1 in m.models:
                    m1.Gaussian_noise.variance.fix(1.0e-8)
                m = GPyMultiOutputWrapper(m, fo.task, n_optimization_restarts=N_Restar)
            elif model_M == 'NAR_con':
                base_kernel = GPy.kern.Matern52
                kernels = make_non_linear_kernels(base_kernel, fo.task, fo.d, 'single1')
                m = NonLinearMultiFidelityModel(Xt, YS, n_fidelities=fo.task, kernels=kernels,verbose=True, optimization_restarts=N_Restar,normalizer=self.normalization)
                for m1 in m.models:
                    m1.Gaussian_noise.variance.fix(1.0e-8)
                # m = GPyMultiOutputWrapper(m, fo.task, n_optimization_restarts=N_Restar)
            elif model_M == 'AR1':
                kernels = [GPy.kern.Matern52(fo.d,ARD=self.ARD)+ GPy.kern.Bias(fo.d), GPy.kern.Matern52(fo.d,ARD=self.ARD)+ GPy.kern.Bias(fo.d)]
                lin_mf_kernel = LinearMultiFidelityKernel(kernels)
                m = GPyLinearMultiFidelityModel(Xt, YS, lin_mf_kernel, n_fidelities=fo.task)
                m.mixed_noise.Gaussian_noise.fix(1.0e-8)
                m.mixed_noise.Gaussian_noise_1.fix(1.0e-8)
                m = GPyMultiOutputWrapper(m, 2, n_optimization_restarts=N_Restar)

            elif model_M == 'Correct':
                m=Correct_model( self.Xt, self.YS, self.fo,  self.normalization,N_Restar)
            elif model_M == 'Con':
                K_con = GPy.kern.Matern52(input_dim=fo.d, active_dims=list(np.arange(0,fo.d,1)),ARD=self.ARD)+ GPy.kern.Bias(fo.d)
                if self.Kint=='Linear':
                    K_int = GPy.kern.Linear(input_dim=1, active_dims=[fo.d])

                elif self.Kint=='MLPL':
                    k1 = GPy.kern.MLP(input_dim=1, active_dims=[fo.d])
                    k2 = GPy.kern.Linear(input_dim=1, active_dims=[fo.d])
                    K_add = GPy.kern.Add([k1, k2])
                    K_pro = GPy.kern.Prod([k1, k2])
                    K_int = GPy.kern.Add([K_add, K_pro])
                else:
                    K_int = GPy.kern.Matern52(input_dim=1, active_dims=[fo.d])




                K_add = GPy.kern.Add([K_int, K_con])
                K_pro = GPy.kern.Prod([K_int, K_con])
                K = GPy.kern.Add([K_add, K_pro])
                m = GPy.models.GPRegression(Xt, YS, kernel=K, normalizer=self.normalization)
                m.Gaussian_noise.fix(1.0e-8)
                m = GPyModelWrapper(m, n_restarts=N_Restar)


            else:
                K2 = GPy.kern.StdPeriodic(fo.d)
                K3 = GPy.kern.Matern52(fo.d)+ GPy.kern.Bias(fo.d)
                lcm = GPy.util.multioutput.LCM(input_dim=fo.d, num_outputs=fo.task, kernels_list=[K2, K3])
                m = GPy.models.GPCoregionalizedRegression(X_train, Y_train, kernel=lcm, normalizer=self.normalization)
                m['.*noise*'].constrain_fixed(1.0e-8)
                m = GPyMultiOutputWrapper(m, fo.task, n_optimization_restarts=N_Restar)
        else:

            if model_M == 'SGP':
                if len(fo.d_int)>0:
                    K_int = GPy.kern.Matern52(input_dim=len(fo.d_int), active_dims=fo.d_int)
                if len(fo.d_float)>0:
                    K_con = GPy.kern.Matern52(input_dim=len(fo.d_float), active_dims=fo.d_float)
                if     len(fo.d_int)>0 and len(fo.d_float)>0:
                    K_add = GPy.kern.Add([K_int, K_con])
                    K_pro = GPy.kern.Prod([K_int, K_con])
                    K = GPy.kern.Add([K_add, K_pro])
                elif len(fo.d_int)>0:
                    K = K_int
                else:
                    K = K_con
                m = GPy.models.GPRegression(X_train[-1], Y_train[-1], kernel=K, normalizer=self.normalization)
                m.Gaussian_noise.fix(1.0e-8)
                m = GPyModelWrapper(m, n_restarts=N_Restar)
                # m = GPyMultiOutputWrapper(m, 1, n_optimization_restarts=N_Restar)

            elif model_M == 'ICM':
                K_int = GPy.kern.Matern52(input_dim=len(fo.d_int), active_dims=fo.d_int)
                K_con = GPy.kern.Matern52(input_dim=len(fo.d_float), active_dims=fo.d_float)
                K_add = GPy.kern.Add([K_int, K_con])
                K_pro = GPy.kern.Prod([K_int, K_con])
                K = GPy.kern.Add([K_add, K_pro])

                icm = GPy.util.multioutput.ICM(input_dim=fo.d, num_outputs=fo.task, kernel=K, W_rank=fo.task)
                m = GPy.models.GPCoregionalizedRegression(X_train, Y_train, kernel=icm, normalizer=self.normalization)
                #m = GPy.models.GPCoregionalizedRegression(X_train, Y_train, kernel=icm)

                m['.*noise*'].constrain_fixed(1.0e-8)
                m = GPyMultiOutputWrapper(m, fo.task, n_optimization_restarts=N_Restar)
            elif model_M == 'NAR':
                base_kernel = GPy.kern.Matern52
                kernels = make_non_linear_kernels(base_kernel, fo.task, fo.d, 'single')
                m = NonLinearMultiFidelityModel(Xt, YS, n_fidelities=fo.task, kernels=kernels, verbose=True,
                                                optimization_restarts=N_Restar,normalizer=self.normalization)
                for m1 in m.models:
                    m1.Gaussian_noise.variance.fix(1.0e-8)
                # m = GPyMultiOutputWrapper(m, fo.task, n_optimization_restarts=N_Restar)

            elif model_M == 'AR1':
                kernels = [GPy.kern.Matern52(fo.d), GPy.kern.Matern52(fo.d)]
                lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
                m = GPyLinearMultiFidelityModel(Xt, YS, lin_mf_kernel, n_fidelities=fo.task)
                m.mixed_noise.Gaussian_noise.fix(1.0e-8)
                m.mixed_noise.Gaussian_noise_1.fix(1.0e-8)
                m = GPyMultiOutputWrapper(m, 2, n_optimization_restarts=N_Restar)
            else:
                K_int = GPy.kern.Matern52(input_dim=len(fo.d_int), active_dims=fo.d_int)
                K_con = GPy.kern.Matern52(input_dim=len(fo.d_float), active_dims=fo.d_float)
                K_add = GPy.kern.Add([K_int, K_con])
                K_pro = GPy.kern.Prod([K_int, K_con])
                K2 = GPy.kern.Add([K_add, K_pro])

                K_int = GPy.kern.StdPeriodic(input_dim=len(fo.d_int), active_dims=fo.d_int)
                K_con = GPy.kern.StdPeriodic(input_dim=len(fo.d_float), active_dims=fo.d_float)
                K_add = GPy.kern.Add([K_int, K_con])
                K_pro = GPy.kern.Prod([K_int, K_con])
                K3 = GPy.kern.Add([K_add, K_pro])

                lcm = GPy.util.multioutput.LCM(input_dim=fo.d, num_outputs=fo.task, kernels_list=[K2, K3])
                m = GPy.models.GPCoregionalizedRegression(X_train, Y_train, kernel=lcm, normalizer=self.normalization)
                m['.*noise*'].constrain_fixed(1.0e-8)
                m = GPyMultiOutputWrapper(m, fo.task, n_optimization_restarts=N_Restar)




        return m,X_train,Y_train





class initial_model_cost:
    def __init__(self,Xt,YS,fo,model_M,normalization=False,d_int=False,ARD=False):
        self.Xt=Xt
        self.YS=YS
        self.fo=fo
        self.model_M=model_M
        self.normalization=normalization
        self.d_int=d_int
        self.ARD = ARD
    def __call__(self):
        Xt=self.Xt
        YS=self.YS
        fo=self.fo
        model_M = self.model_M
        N_Restar=1

        K_con = GPy.kern.Matern52(input_dim=fo.d, active_dims=list(np.arange(0,fo.d,1)),ARD=self.ARD)+ GPy.kern.Bias(fo.d)
        K_int = GPy.kern.Linear(input_dim=1, active_dims=[fo.d])
        K_add = GPy.kern.Add([K_int, K_con])
        K_pro = GPy.kern.Prod([K_int, K_con])
        K = GPy.kern.Add([K_add, K_pro])
        m = GPy.models.GPRegression(Xt, YS, kernel=K, normalizer=self.normalization)
        m.Gaussian_noise.fix(1.0e-8)
        m = GPyModelWrapper(m, n_restarts=N_Restar)


        return m



class Correct_model:
    def __init__(self, Xt, YS, fo, normalization=False,N_Restar=1):
        self.Xt = Xt
        self.YS = YS
        self.fo = fo
        self.normalization = normalization
        self.N_Restar = N_Restar

        self.n_outputs=self.fo.task
        self.Y= YS
        self.X= Xt
    def set_data(self,X,Y):
        self.Xt = X
        self.YS = Y
        self.Y= Y
        self.X= X

    def optimize(self):
        para=[]
        X_train, Y_train = [],[]
        for i in range(self.fo.task):
            X_train.append(self.Xt[self.Xt[:, -1] == i, :-1])
            Y_train.append(self.YS[self.Xt[:, -1] == i, :])
        #  LF GP model
        K_LF = GPy.kern.Matern52(self.fo.d)
        m_LF = GPy.models.GPRegression(X_train[0], Y_train[0], kernel=K_LF, normalizer=self.normalization)
        m_LF.Gaussian_noise.fix(1.0e-8)
        m_LF = GPyMultiOutputWrapper(m_LF, 1, n_optimization_restarts=self.N_Restar)
        m_LF.optimize()
        x,v = m_LF.predict(X_train[-1])

        y = Y_train[-1]

        reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
        pre = reg.predict(x.reshape(-1, 1))

        error = y - pre

        K_error = GPy.kern.Matern52(self.fo.d)
        m_error = GPy.models.GPRegression(X_train[-1], error.reshape(-1, 1), kernel=K_error, normalizer=self.normalization)
        m_error.Gaussian_noise.fix(1.0e-8)
        m_error = GPyMultiOutputWrapper(m_error, 1, n_optimization_restarts=self.N_Restar)
        m_error.optimize()
        self.m_LF=m_LF
        self.m_error=m_error

        self.a = reg.coef_[0]
        self.b = reg.intercept_
        self.reg=reg
        para.append({"a": self.a.tolist(), "b": self.b.tolist()})
    def predict(self,x):
        ind=x[:,-1]==0

        mean_x,var_x=np.ones((x.shape[0],1)),np.ones((x.shape[0],1))
        if len((x[ind,:-1]))>0:
            mean_lf, var_lf = self.m_LF.predict(x[ind,:-1])
            # pre_lf = self.reg.predict(mean_lf)
            # mean = mean_error + pre_lf
            # var = var_lf + var_error
            mean_x[ind,:]=mean_lf
            var_x[ind,:]=var_lf

        ind=x[:,-1]==1
        if len((x[ind,:-1]))>0:
            mean_error, var_error = self.m_error.predict(x[ind,:-1])
            mean_lf, var_lf = self.m_LF.predict(x[ind,:-1])
            pre_lf =self.reg.predict(mean_lf)
            mean = mean_error + pre_lf
            var = var_error
            mean_x[ind,:]=mean
            var_x[ind,:]=var
        return mean_x,var_x


