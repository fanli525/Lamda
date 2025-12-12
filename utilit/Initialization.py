
import GPy
import emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class initial:
    def __init__(self,N_train,maxN,FM_name,AC,model_M,lh_threshold,c_threshold,r):
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
    def __init__(self,Xt,YS,fo,model_M,normalization=False,d_int=False):
        self.Xt=Xt
        self.YS=YS
        self.fo=fo
        self.model_M=model_M
        self.normalization=normalization

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


        if len(self.d_int)==False:
            if model_M == 'SGP':
                K = GPy.kern.Matern52(fo.d)
                m = GPy.models.GPRegression(X_train[-1], Y_train[-1], kernel=K,normalizer=self.normalization)
                m.Gaussian_noise.fix(1.0e-8)
                m = GPyMultiOutputWrapper(m, 1, n_optimization_restarts=N_Restar)
            elif model_M == 'ICM':
                K = GPy.kern.Matern52(fo.d)
                icm = GPy.util.multioutput.ICM(input_dim=fo.d, num_outputs=fo.task, kernel=K, W_rank=fo.task)
                m = GPy.models.GPCoregionalizedRegression(X_train, Y_train, kernel=icm,normalizer=self.normalization)
                m['.*noise*'].constrain_fixed(1.0e-8)
                m = GPyMultiOutputWrapper(m, fo.task, n_optimization_restarts=N_Restar)
            elif model_M == 'NAR':
                base_kernel = GPy.kern.Matern52
                kernels = make_non_linear_kernels(base_kernel, fo.task, fo.d, 'single')
                m = NonLinearMultiFidelityModel(Xt, YS, n_fidelities=fo.task, kernels=kernels,verbose=True, optimization_restarts=N_Restar)
                for m1 in m.models:
                    m1.Gaussian_noise.variance.fix(1.0e-8)
            elif model_M == 'AR1':
                kernels = [GPy.kern.Matern52(fo.d), GPy.kern.Matern52(fo.d)]
                lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
                m = GPyLinearMultiFidelityModel(Xt, YS, lin_mf_kernel, n_fidelities=fo.task)
                m.mixed_noise.Gaussian_noise.fix(1.0e-8)
                m.mixed_noise.Gaussian_noise_1.fix(1.0e-8)
                m = GPyMultiOutputWrapper(m, 2, n_optimization_restarts=N_Restar)
            else:
                K2 = GPy.kern.StdPeriodic(fo.d)
                K3 = GPy.kern.Matern52(fo.d)
                lcm = GPy.util.multioutput.LCM(input_dim=fo.d, num_outputs=fo.task, kernels_list=[K2, K3])
                m = GPy.models.GPCoregionalizedRegression(X_train, Y_train, kernel=lcm, normalizer=self.normalization)
                m['.*noise*'].constrain_fixed(1.0e-8)
                m = GPyMultiOutputWrapper(m, fo.task, n_optimization_restarts=N_Restar)
        else:

            if model_M == 'SGP':

                K_int = GPy.kern.Matern52(input_dim=len(fo.d_int), active_dims=fo.d_int)
                K_con = GPy.kern.Matern52(input_dim=len(fo.d_no_int), active_dims=fo.d_no_int)
                K_add = GPy.kern.Add([K_int, K_con])
                K_pro = GPy.kern.Prod([K_int, K_con])
                K = GPy.kern.Add([K_add, K_pro])
                m = GPy.models.GPRegression(X_train[-1], Y_train[-1], kernel=K, normalizer=self.normalization)
                m.Gaussian_noise.fix(1.0e-8)
                m = GPyMultiOutputWrapper(m, 1, n_optimization_restarts=N_Restar)
            elif model_M == 'ICM':
                K_int = GPy.kern.Matern52(input_dim=len(fo.d_int), active_dims=fo.d_int)
                K_con = GPy.kern.Matern52(input_dim=len(fo.d_no_int), active_dims=fo.d_no_int)
                K_add = GPy.kern.Add([K_int, K_con])
                K_pro = GPy.kern.Prod([K_int, K_con])
                K = GPy.kern.Add([K_add, K_pro])
                icm = GPy.util.multioutput.ICM(input_dim=fo.d, num_outputs=fo.task, kernel=K, W_rank=fo.task)
                m = GPy.models.GPCoregionalizedRegression(X_train, Y_train, kernel=icm, normalizer=self.normalization)
                m['.*noise*'].constrain_fixed(1.0e-8)
                m = GPyMultiOutputWrapper(m, fo.task, n_optimization_restarts=N_Restar)
            elif model_M == 'NAR':
                base_kernel = GPy.kern.Matern52
                kernels = make_non_linear_kernels(base_kernel, fo.task, fo.d, 'single')
                m = NonLinearMultiFidelityModel(Xt, YS, n_fidelities=fo.task, kernels=kernels, verbose=True,
                                                optimization_restarts=N_Restar)
                for m1 in m.models:
                    m1.Gaussian_noise.variance.fix(1.0e-8)
            elif model_M == 'AR1':
                kernels = [GPy.kern.Matern52(fo.d), GPy.kern.Matern52(fo.d)]
                lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
                m = GPyLinearMultiFidelityModel(Xt, YS, lin_mf_kernel, n_fidelities=fo.task)
                m.mixed_noise.Gaussian_noise.fix(1.0e-8)
                m.mixed_noise.Gaussian_noise_1.fix(1.0e-8)
                m = GPyMultiOutputWrapper(m, 2, n_optimization_restarts=N_Restar)
            else:
                K_int = GPy.kern.Matern52(input_dim=len(fo.d_int), active_dims=fo.d_int)
                K_con = GPy.kern.Matern52(input_dim=len(fo.d_no_int), active_dims=fo.d_no_int)
                K_add = GPy.kern.Add([K_int, K_con])
                K_pro = GPy.kern.Prod([K_int, K_con])
                K2 = GPy.kern.Add([K_add, K_pro])

                K_int = GPy.kern.StdPeriodic(input_dim=len(fo.d_int), active_dims=fo.d_int)
                K_con = GPy.kern.StdPeriodic(input_dim=len(fo.d_no_int), active_dims=fo.d_no_int)
                K_add = GPy.kern.Add([K_int, K_con])
                K_pro = GPy.kern.Prod([K_int, K_con])
                K3 = GPy.kern.Add([K_add, K_pro])

                lcm = GPy.util.multioutput.LCM(input_dim=fo.d, num_outputs=fo.task, kernels_list=[K2, K3])
                m = GPy.models.GPCoregionalizedRegression(X_train, Y_train, kernel=lcm, normalizer=self.normalization)
                m['.*noise*'].constrain_fixed(1.0e-8)
                m = GPyMultiOutputWrapper(m, fo.task, n_optimization_restarts=N_Restar)




        return m,X_train,Y_train