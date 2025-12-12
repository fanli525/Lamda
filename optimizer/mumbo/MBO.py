import time, random
import numpy as np
from optimizer.utilit.init_latin_hypercube_sampling import init_latin_hypercube_sampling
from optimizer.utilit.Initialization_mix import initial_model, initial_model_cost
from scipy.spatial.distance import cdist
from emukit.bayesian_optimization.acquisitions.max_value_entropy_search import MUMBO
from emukit.core.acquisition import Acquisition
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.loop.candidate_point_calculators import GreedyBatchPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

import copy, os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'







class Cost_fun(Acquisition):
    def __init__(self, mc):
        self.mc = mc

    def evaluate(self, x):
        mu, var = self.mc.predict(x)

        return mu

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evalute(x), np.zeros(x.shape)


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
        xint = init_latin_hypercube_sampling(np.array(Tbd), np.array(Tbu), Nt, np.random.RandomState(Param.r))
        per_t = int(np.ceil(Nt / len(fo.x_fid)))
        if len(Param.prior) == 0:
            for i in range(len(fo.x_fid)):
                xint[i * per_t:(i + 1) * per_t, -1] = fo.x_fid[i]
        else:
            xint[:, -1] = fo.x_fid[-1]

        if len(Param.prior) > 0:
            X_prior = np.array(Param.prior[0]['Samp'])
            Y_prior = Param.prior[0]['YS']
            C_prior = Param.prior[0]['cost_all']
            Param.bd_prom = Param.prior[0]['bd_prom']
            xint = np.vstack((X_prior, xint))
        Xt = xint.copy()
        Xt1 = xint.copy()
        if len(Param.prior) == 0:
            for i in range(len(fo.x_fid)):
                Xt[i * per_t:(i + 1) * per_t, -1] = i
        else:
            Xt[:-Nt, -1] = 0
            Xt[-Nt:, -1] = 1

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
        cost_HF = np.mean(cost[Xt1[:, -1] == fo.x_fid[-1]])
        if cost_HF < 10:
            maxcost = 5 * Param.maxN * cost_HF
        else:
            maxcost = max(Param.maxN * cost_HF, 10000)
        maxcost = Param.maxN * cost_HF

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
        modelc = initial_model_cost(Xt, YC, fo, model_M, normalization, d_int=False)
        # model=initial_model(Xt,YS,fo,model_M,normalization,d_int=fo.d_int)
        m, X_train, Y_train = model()
        # mc=modelc()
        it, npv, p0 = 0, 0, 0.8
        batch_size = 3
        if len(Param.prior) > 0:
            bd_prom = Param.bd_prom.copy()
        else:
            Param.bd_prom, bd_prom = [], []
        if len(Param.bd_prom) > 0:
            if len(fo.d_log) > 0:
                bd_prom[fo.d_log, :] = np.log(bd_prom[fo.d_log, :])
        NFill = 0
        x1, x2 = X_train[0], X_train[1]
        y1, y2 = Y_train[0], Y_train[1]
        ratios = [int(fo.x_fid[-1] / x) for x in fo.x_fid]
        usage_order = []
        for i, count in enumerate(ratios):
            usage_order.extend([fo.x_fid[i]] * count)

        # 在 maxit 循环中依次使用 usage_order 的值，并在每一轮重复
        order_length = len(usage_order)
        while runtime[-1] < maxcost and NFill < 30:
            if fo.task == 2:
                if (it + 1) % int(fo.x_fid[1] / fo.x_fid[0] + 1) != 0:
                    ind_task = 0
                else:
                    ind_task = 1
            else:
                current_x_fid = usage_order[it % order_length]
                ind_task = fo.x_fid.index(current_x_fid)
            print(model_M, "runtime/maxcos: ", runtime[-1], '/', maxcost, 'y_opt: ', np.min(Ymin))
            if norm_Y == 'log':
                model = initial_model(Xt, np.log(YS), fo, model_M, normalization, d_int=False, Kint=Param.Kint)
            else:
                model = initial_model(Xt, YS, fo, model_M, normalization, d_int=False, Kint=Param.Kint)
            m, X_train, Y_train = model()
            m.optimize()
            p1 = p0 ** (it)
            from optimizer.mumbo.emukit.bayesian_optimization.acquisitions.MEI import per_imLCB, per_MB, per_imEI

            high_gp_model1, var_H = [], []
            n_obj = 2
            LCB_w = 2
            fun_LCB = per_imLCB(m, N_train, cost, dimension, LCB_w, high_gp_model1, ind_task, var_H)
            # if Param.AC=='MB':
            problem = per_MB(n_var=fo.d, n_obj=n_obj, xl=Tbd[:-1], xu=Tbu[:-1], model=m, N_train=N_train, cost=cost,
                             d=dimension, w=2,
                             high_gp_model=[], ind_task=ind_task, var_H=0)
            from pymoo.optimize import minimize
            from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
            t1 = time.time()
            algorithm = NSGA2(pop_size=100)
            res = minimize(problem, algorithm, ('n_gen', 10), seed=1, verbose=False)
            t2 = time.time()
            print('nsga2 time:', t2 - t1)
            Xc = res.X
            Yc = res.F
            nc = np.min([batch_size, Xc.shape[0]])
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=nc, random_state=0).fit(Xc)
            label = kmeans.labels_


            X_new1, Ys_new = [], []
            for i in range(nc):
                x_new = Xc[label == i, :]
                if x_new.shape[0] > 0:
                    c = fun_LCB(x_new)
                    c = -Yc[label == i, 0]
                    sorted_id = sorted(range(len(c)), key=lambda k: c[k], reverse=True)
                    index = sorted_id[0]
                    x_new0 = np.atleast_2d(x_new[index, :])
                    Ys_new.append(Yc[label == i, :][index])
                    if len(X_new1) < 1:
                        X_new1 = x_new0
                    else:
                        X_new1 = np.vstack((X_new1, x_new0))
            next_mb=X_new1
            if Param.AC=='EI':
                fun_EI = per_imEI(m, N_train, cost, dimension, high_gp_model1, ind_task, var_H)
                optparams = differential_evolution(fun_EI, bounds=problem_bounds, maxiter=500)
                next_ei = optparams.x
                X_new1 = np.atleast_2d(next_ei)
            elif Param.AC=='LCB':
                fun_LCB = per_imLCB(m, N_train, cost, dimension, LCB_w, high_gp_model1, ind_task, var_H)
                optparams = differential_evolution(fun_LCB, bounds=problem_bounds, maxiter=500)
                next_lcb = optparams.x
                X_new1 = np.atleast_2d(next_lcb)
            elif Param.AC=='MES':
                modelc = initial_model_cost(Xt, YC, fo, model_M, normalization, d_int=False)
                mc = modelc()
                space = [ContinuousParameter('x' + str(i), Tbd[i], Tbu[i]) for i in range(fo.d)]
                s2 = space.append(InformationSourceParameter(fo.task))
                s2 = space
                parameter_space = ParameterSpace(s2)
                cost_acquisition = Cost_fun(mc)
                mumbo_acquisition = MUMBO(m, parameter_space, num_samples=10, grid_size=500 * fo.d) / cost_acquisition
                acquisition = mumbo_acquisition
                initial_loop_state = create_loop_state(Xt, np.log(YS))
                acquisition_optimizer = MultiSourceAcquisitionOptimizer(
                    GradientAcquisitionOptimizer(parameter_space, num_anchor=3),
                    parameter_space)
                candidate_point_calculator = GreedyBatchPointCalculator(m, acquisition, acquisition_optimizer, batch_size=1)
                new_mes = candidate_point_calculator.compute_next_points(initial_loop_state)
                X_new1 = new_mes[:, :-1]
            X_new = X_new1
            for i in range(len(X_new)):
                x_new = np.atleast_2d(X_new[i, :])
                print('x_new:', x_new)
                fun_LCB(x_new)
                if fun_LCB.sigma[0, 0] > 0.01:
                    x_new1, x_new2 = np.zeros((1, fo.d + 1)), np.zeros((1, fo.d + 1))
                    x_new1[:, :-1], x_new2[:, :-1] = x_new.copy(), x_new.copy()
                    x_new2[:, -1] = fo.x_fid[ind_task]
                    x_new1[:, -1] = ind_task
                    dist = cdist(x_new2, Xt1, metric='euclidean')
                    if np.min(dist) > dalta:
                        NFill = 0
                        if len(fo.d_log) > 0:
                            x_new2[:, fo.d_log] = np.exp(x_new2[:, fo.d_log])
                        if len(fo.d_int) > 0:
                            x_new2[:, fo.d_int] = np.round(x_new2[:, fo.d_int])
                        # x_new[:,-1] =  np.round(np.exp(x_new[:,-1]))
                        x_new2[0, :-1][(x_new2[0, :-1] > np.array(fo.bound)[:, 1])] = np.array(fo.bound)[:, 1][
                            (x_new2[0, :-1] > np.array(fo.bound)[:, 1])]
                        x_new2[0, :-1][(x_new2[0, :-1] < np.array(fo.bound)[:, 0])] = np.array(fo.bound)[:, 0][
                            (x_new2[0, :-1] < np.array(fo.bound)[:, 0])]
                        y_new, c_new = fo.fit(x_new2[0, :-1], x_new2[0, -1])
                        x_fid = int(x_new2[:, -1])
                        print('x_fid: ', x_fid, ' y_new:', y_new)
                        Xt = np.vstack((Xt, x_new1))
                        Samp = np.vstack((Samp, x_new2))
                        YS = np.append(YS, y_new)
                        YC = np.append(YC, c_new)
                        YS = YS[:, np.newaxis]
                        YC = YC[:, np.newaxis]
                        cost_all = np.append(cost_all, c_new)
                        if x_new2[0][-1] == fo.bound_fid[1]:
                            if inc_valid > y_new:
                                inc_valid = y_new
                        Ymin.append(float(inc_valid))
                        rt += cost_all[-1]
                        runtime.append(float(rt))
                        fo.runtime = runtime
                        Param.Ymin = Ymin
                else:
                    NFill += 1

            it = it + 1
            # 绘制二维点的分布
            if Param.plt_ac==True:
                fun_EI = per_imEI(m, N_train, cost, dimension, high_gp_model1, ind_task, var_H)

                optparams = differential_evolution(fun_EI, bounds=problem_bounds, maxiter=500)
                next_ei = optparams.x
                y_ei, s_ei = fun_EI.pre[0, 0], -fun_EI.sigma[0, 1]

                optparams = differential_evolution(fun_LCB, bounds=problem_bounds, maxiter=500)
                next_lcb = optparams.x
                y_lcb, s_lcb = fun_LCB.pre[0, 0], -fun_LCB.sigma[0, 1]

                modelc = initial_model_cost(Xt, YC, fo, model_M, normalization, d_int=False)
                mc = modelc()
                space = [ContinuousParameter('x' + str(i), Tbd[i], Tbu[i]) for i in range(fo.d)]
                s2 = space.append(InformationSourceParameter(fo.task))
                s2 = space
                parameter_space = ParameterSpace(s2)
                cost_acquisition = Cost_fun(mc)
                mumbo_acquisition = MUMBO(m, parameter_space, num_samples=10, grid_size=500 * fo.d) / cost_acquisition
                acquisition = mumbo_acquisition
                initial_loop_state = create_loop_state(Xt, np.log(YS))
                acquisition_optimizer = MultiSourceAcquisitionOptimizer(
                    GradientAcquisitionOptimizer(parameter_space, num_anchor=3),
                    parameter_space)
                candidate_point_calculator = GreedyBatchPointCalculator(m, acquisition, acquisition_optimizer, batch_size=1)
                new_mes = candidate_point_calculator.compute_next_points(initial_loop_state)
                fun_EI(new_mes[0, :-1])
                y_mes, s_mes = fun_EI.pre[0, 0], -fun_EI.sigma[0, 1]
                Ys_new = np.array(y_mes)


                grid_data = ('', '', '')
                markers = [(Yc, 'blue', 'o'),
                           (Ys_new, 'yellow', 's'),
                            (np.array([y_lcb, s_lcb]).reshape(1,-1), 'red', '*'),
                           (np.array([y_ei, s_ei]).reshape(1,-1), 'green', '*'),
                           (np.array([y_mes, s_mes]).reshape(1,-1), 'orange', '*')]
                # from utils.plt_2D import Args, plt_2D, save_data_to_dat
                # save_data_to_dat(output_path='./output', name='MB_pf' + str(it), markers=Yc)
                # save_data_to_dat(output_path='./output', name='MB_final' + str(it), markers=Ys_new)
                # save_data_to_dat(output_path='./output', name='MB_LCB' + str(it), markers=np.array([y_lcb, s_lcb]).reshape(1,-1))
                # save_data_to_dat(output_path='./output', name='MB_EI' + str(it), markers=np.array([y_ei, s_ei]).reshape(1,-1))
                # save_data_to_dat(output_path='./output', name='MB_MES' + str(it), markers=np.array([y_mes, s_mes]).reshape(1,-1))


                # 设置 markers
                # args = Args(
                #     x_label='$\mu$',  # 新的 x_label
                #     y_label='$\sigma$',
                #     output_path='./output',
                #     name='pf_plot_iter'+str(it),
                # )
                # plt_2D(grid_data, args, markers)




                #  plot meshgrid of the prediction and sigma
                if fo.d==1:
                    x1 = np.linspace(problem_bounds[0][0], problem_bounds[0][1], 100)
                    Xtest = x1.reshape(-1, 1)
                    ind = np.ones((Xtest.shape[0], 1))
                    Xtest = np.hstack((Xtest, ind))
                    ind0 = np.zeros((Xtest.shape[0], 1))
                    Xtest0 = np.hstack((x1.reshape(-1, 1), ind0))
                    mean0, var0 = m.predict(Xtest0)
                    mean0 = mean0.reshape(Xtest0.shape[0])

                    mean, var = m.predict(Xtest)
                    mean = mean.reshape(Xtest.shape[0])
                    Xtest1 = copy.deepcopy(Xtest)

                    if len(fo.d_log) > 0:
                        Xtest[:, fo.d_log] = np.exp(Xtest[:, fo.d_log])
                    if len(fo.d_int) > 0:
                        Xtest[:, fo.d_int] = np.round(Xtest[:, fo.d_int])
                    Xtest[:, -1] = fo.x_fid[-1]
                    Xtest_l=copy.deepcopy(Xtest)
                    Xtest_l[:, -1] = fo.x_fid[0]
                    Ytest = np.zeros((Xtest.shape[0], 1))
                    Ytest_l = np.zeros((Xtest.shape[0], 1))
                    for i in range(Xtest.shape[0]):
                        y, c = fo.fit(Xtest[i, :-1], Xtest[i, -1])
                        Ytest[i] = y
                        y1, c1 = fo.fit(Xtest_l[i, :-1], Xtest_l[i, -1])
                        Ytest_l[i] = y1
                    Ytest = Ytest.reshape(Xtest.shape[0])
                    Ytest_l = Ytest_l.reshape(Xtest.shape[0])
                    fun_LCB.ind_task = 1
                    fun_EI.ind_task = 1
                    lcb_xtest=fun_LCB(Xtest1[:,0]).reshape(Xtest.shape[0],-1)
                    ei_xtest=fun_EI(Xtest1[:,0]).reshape(Xtest.shape[0],-1)
                    mes_xtest=mumbo_acquisition.evaluate(Xtest1)

                    grid_data = ('', '', '')
                    markers = [(Xt, 'k', 'o'),
                               (next_mb, 'yellow', 's'),
                               (next_lcb.reshape(1, -1), 'c', 'd'),
                               (next_ei.reshape(1, -1), 'green', '^'),
                               (new_mes, 'orange', '^'),
                               (np.vstack((x1,lcb_xtest[:,-1])).T, 'c', '.'),
                               (np.vstack((x1, ei_xtest[:, -1])).T, 'green', '.'),
                                (np.vstack((x1, mes_xtest[:, -1])).T, 'orange', '.')]


                    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

                    # First subplot with lcb, ei, mes, and sampling points
                    axs[0].plot(x1, lcb_xtest[:, -1], 'c', label='lcb')
                    axs[0].plot(x1, ei_xtest[:, -1], 'green', label='ei')
                    axs[0].plot(x1, mes_xtest[:, -1], 'orange', label='mes')
                    axs[0].plot(next_mb, 0.1 * np.ones_like(next_mb), 'o', color='yellow', label='mb')
                    axs[0].plot(next_lcb, 0.2 * np.ones_like(next_lcb), 'o', color='c', label='lcb')
                    axs[0].plot(next_ei, 0.3 * np.ones_like(next_ei), 'o', color='green', label='ei')
                    axs[0].plot(new_mes[:, 0], 0.4 * np.ones_like(new_mes), 'o', color='orange', label='mes')
                    axs[0].legend()
                    axs[0].set_title('Model Predictions with Sampling Points. Iteration: ' + str(it))
                    axs[0].set_xlabel('x')
                    axs[0].set_ylabel('Prediction Values')
                    output_path = './output_ac'
                    if not os.path.exists(output_path):
                        os.makedirs(output_path, exist_ok=True)
                    name = 'ac_it' + str(it)+'.dat'
                    file_out = os.path.join(output_path, name)
                    np.savetxt(file_out, np.column_stack((x1, lcb_xtest[:, -1], -ei_xtest[:, -1],
                                                                   mes_xtest[:, -1])),
                               header="x1 lcb ei mes", comments='')


                    name = 'ac_n_it' + str(it)+'.dat'
                    file_out = os.path.join(output_path, name)

                    ei_n=(-ei_xtest[:, -1]-np.min(-ei_xtest[:, -1]))/(np.max(-ei_xtest[:, -1])-np.min(-ei_xtest[:, -1]))
                    mes_n=(mes_xtest[:, -1]-np.min(mes_xtest))/(np.max(mes_xtest)-np.min(mes_xtest))


                    np.savetxt(file_out, np.column_stack((x1, lcb_xtest[:, -1], ei_n,
                                                                   mes_n)),
                               header="x1 lcb ei mes", comments='')




                    name = 'mb_it' + str(it)+'.dat'
                    file_out = os.path.join(output_path, name)
                    np.savetxt(file_out, np.column_stack(( next_mb)),
                               header="mb", comments='')

                    name = 'lcb_it' + str(it)+'.dat'
                    file_out = os.path.join(output_path, name)
                    np.savetxt(file_out, np.column_stack((  next_lcb)),
                               header="lcb", comments='')

                    name = 'ei_it' + str(it)+'.dat'
                    file_out = os.path.join(output_path, name)
                    np.savetxt(file_out, np.column_stack(( next_ei)),
                               header="ei", comments='')

                    name = 'mes_it' + str(it)+'.dat'
                    file_out = os.path.join(output_path, name)
                    np.savetxt(file_out, np.column_stack(( new_mes[:, 0])),
                               header="mes", comments='')

                    # Second subplot with real data and sampled points
                    axs[1].plot(Xtest[:, 0], Ytest, 'c', label='real')
                    s1=np.sqrt(var)
                    axs[1].fill_between(Xtest[:, 0].flatten(), (mean - 0.96* s1[:, 0]).flatten(),
                                     (mean + 0.96* s1[:, 0]).flatten(), facecolor='g', alpha=0.3)
                    axs[1].plot(Xt[Xt[:, -1] == 1, 0], YS[Xt[:, -1] == 1, 0], 'o', color='r', label='sample')
                    axs[1].legend()
                    axs[1].set_title('Real Function and Sampled Points. Iteration: ' + str(it))
                    axs[1].set_xlabel('x')
                    axs[1].set_ylabel('Function Value')


                    name = 'hf_it' + str(it)+'.dat'
                    file_out = os.path.join(output_path, name)
                    np.savetxt(file_out, np.column_stack((Xtest[:, 0], Ytest,mean,
                                                                   (mean - 0.96 * s1[:, 0]),
                                                                   (mean + 0.96 * s1[:, 0]))),
                               header="x y m lb ub",
                               comments='')

                    name = 'hf_train_it' + str(it)+'.dat'
                    file_out = os.path.join(output_path, name)
                    np.savetxt(file_out, np.column_stack((
                                                                   Xt[Xt[:, -1] == 1, 0], YS[Xt[:, -1] == 1, 0])),
                               header="x y",
                               comments='')


                    # 保存第三张图的数据

                    axs[2].plot(Xtest[:, 0], Ytest_l, 'b', label='real lf')
                    s1=np.sqrt(var0)
                    axs[2].fill_between(Xtest[:, 0].flatten(), (mean0 - 0.96* s1[:, 0]).flatten(),
                                     (mean0 + 0.96* s1[:, 0]).flatten(), facecolor='m', alpha=0.3)
                    axs[2].plot(Xt[Xt[:, -1] == 0, 0], YS[Xt[:, -1] == 0, 0], 'o', color='m', label='sample')


                    axs[2].legend()
                    axs[2].set_title('Real Function and Sampled Points. Iteration: ' + str(it))
                    axs[2].set_xlabel('x')
                    axs[2].set_ylabel('Function Value')
                    plt.tight_layout()
                    plt.show()
                    s1_0 = np.sqrt(var0)
                    name = 'lf_it' + str(it) + '.dat'
                    file_out = os.path.join(output_path, name)
                    np.savetxt(file_out, np.column_stack((Xtest[:, 0], Ytest_l,mean0,
                                                          (mean0 - 0.96 * s1_0[:, 0]),
                                                          (mean0 + 0.96 * s1_0[:, 0]))),
                               header="x y m lb ub",
                               comments='')
                    name = 'lf_train_it' + str(it) + '.dat'
                    file_out = os.path.join(output_path, name)
                    np.savetxt(file_out, np.column_stack((
                        Xt[Xt[:, -1] == 0, 0], YS[Xt[:, -1] == 0, 0])),
                               header="x y",
                               comments='')




                    #
                    # # 设置 markers
                    # from utils.plt_2D import Args, plt_2D
                    # args = Args(
                    #     x_label='$x_1$',  # 新的 x_label
                    #     y_label='$x_2$',
                    #     output_path='./output',
                    #     name='contour_plot_iter' + str(it),
                    # )
                    # plt_2D(grid_data, args, markers)
                    #
                    #
                    # if Param.plt_clu == True:
                    #     save_data_to_dat(output_path='./output', name='MB_pf0' + str(it), markers=Yc[label == 0])
                    #     save_data_to_dat(output_path='./output', name='MB_pf1' + str(it), markers=Yc[label == 1])
                    #     save_data_to_dat(output_path='./output', name='MB_pf2' + str(it), markers=Yc[label == 2])
                    #     save_data_to_dat(output_path='./output', name='MB_final' + str(it), markers=Ys_new)

                else:
                    x1 = np.linspace(problem_bounds[0][0], problem_bounds[0][1], 100)
                    x2 = np.linspace(problem_bounds[1][0], problem_bounds[1][1], 100)
                    X1, X2 = np.meshgrid(x1, x2)
                    Xtest = np.vstack((X1.flatten(), X2.flatten())).T
                    ind=np.ones((Xtest.shape[0],1))
                    Xtest = np.hstack((Xtest,ind))
                    mean, var = m.predict(Xtest)
                    mean = mean.reshape(X1.shape)

                    if len(fo.d_log) > 0:
                        Xtest[:, fo.d_log] = np.exp(Xtest[:, fo.d_log])
                    if len(fo.d_int) > 0:
                        Xtest[:, fo.d_int] = np.round(Xtest[:, fo.d_int])
                    Xtest[:, -1] = fo.x_fid[-1]
                    Ytest=np.zeros((Xtest.shape[0],1))
                    for i in range(Xtest.shape[0]):
                        y, c = fo.fit(Xtest[i, :-1], Xtest[i, -1])
                        Ytest[i]=y
                    Ytest=Ytest.reshape(X1.shape)


                    grid_data = (X1, X2, Ytest)
                    markers = [(Xt, 'k', 'o'),
                                (next_mb, 'yellow', 's'),
                                (next_lcb.reshape(1,-1), 'c', 'd'),
                               (next_ei.reshape(1,-1), 'green', '^'),
                               (new_mes, 'orange', '^')]
                    # 设置 markers
                    from utils.plt_2D import Args, plt_2D
                    args = Args(
                        x_label='$x_1$',  # 新的 x_label
                        y_label='$x_2$',
                        output_path='./output',
                        name='contour_plot_iter'+str(it),
                    )
                    plt_2D(grid_data, args, markers)

                    grid_data = ('', '', '')
                    markers = [(Xt[Xt[:,-1]==0,:], 'k', 'o'),
                                (Xt[Xt[:,-1]==1,:], 'c', 'o')]
                    # 设置 markers
                    args = Args(
                        x_label='$x_1$',  # 新的 x_label
                        y_label='$x_2$',
                        output_path='./output',
                        name='initial_point'+str(it),
                    )
                    plt_2D(grid_data, args, markers)
                    if Param.plt_clu == True:
                        save_data_to_dat(output_path='./output', name='MB_pf0' + str(it), markers=Yc[label == 0])
                        save_data_to_dat(output_path='./output', name='MB_pf1' + str(it), markers=Yc[label == 1])
                        save_data_to_dat(output_path='./output', name='MB_pf2' + str(it), markers=Yc[label == 2])
                        save_data_to_dat(output_path='./output', name='MB_final' + str(it), markers=Ys_new)
                        grid_data = (X1, X2, Ytest)
                        markers = [(Xc[label ==0, :], 'b', 'o'),
                                     (Xc[label ==1, :], 'c', 'o'),
                                        (Xc[label ==2, :], 'purple', 'o'),
                                   (next_mb, 'yellow', 's')]

                        args = Args(
                            x_label='$x_1$',  # 新的 x_label
                            y_label='$x_2$',
                            output_path='./output',
                            name='contour_ac_iter' + str(it),
                        )
                        plt_2D(grid_data, args, markers,plt_opt=False)

                        kmeans1 = KMeans(n_clusters=nc, random_state=0).fit(Yc)
                        label1 = kmeans1.labels_

                        X_new1_xc, Ys_new_xc = [], []
                        for i in range(nc):
                            x_new = Xc[label1 == i, :]
                            if x_new.shape[0] > 0:
                                c = fun_LCB(x_new)
                                c = -Yc[label1 == i, 0]
                                sorted_id = sorted(range(len(c)), key=lambda k: c[k], reverse=True)
                                index = sorted_id[0]
                                x_new0 = np.atleast_2d(x_new[index, :])
                                Ys_new_xc.append(Yc[label1 == i, :][index])
                                if len(X_new1_xc) < 1:
                                    X_new1_xc = x_new0
                                else:
                                    X_new1_xc = np.vstack((X_new1_xc, x_new0))

                        save_data_to_dat(output_path='./output', name='MB_pf0_xc' + str(it), markers=Yc[label1 == 0])
                        save_data_to_dat(output_path='./output', name='MB_pf1_xc' + str(it), markers=Yc[label1 == 1])
                        save_data_to_dat(output_path='./output', name='MB_pf2_xc' + str(it), markers=Yc[label1 == 2])
                        save_data_to_dat(output_path='./output', name='MB_final_xc' + str(it), markers=np.array(Ys_new_xc))
                        grid_data = (X1, X2, Ytest)
                        markers = [(Xc[label1 ==0, :], 'b', 'o'),
                                     (Xc[label1 ==1, :], 'c', 'o'),
                                        (Xc[label1 ==2, :], 'purple', 'o'),
                                   (np.array(X_new1_xc), 'yellow', 's')]

                        args = Args(
                            x_label='$x_1$',  # 新的 x_label
                            y_label='$x_2$',
                            output_path='./output',
                            name='contour_ac_xc_iter' + str(it),
                        )
                        plt_2D(grid_data, args, markers,plt_opt=False)

        T2 = time.time()
        y_opt = min(YS[Samp[:, -1] == Tbu[-1]])
        Ib = np.where(YS == y_opt)[0][0]
        x_opt = Samp[Ib, :-1]

        EV_inf.append({
            "xopt": x_opt.tolist(), "yopt": y_opt.tolist(),
            "Samp": Samp.tolist(), "YS": YS.tolist(),
            "cost_all": cost_all.tolist(), "runtime": runtime,
            "Ymin": Ymin, 'Time': T2 - T1,
        })
        return EV_inf
