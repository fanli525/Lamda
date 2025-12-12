
import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
import argparse
import os, pickle
from P_fcnet  import P_NN,get_hyperparameter_search_space,get_vartypes
import random
from yahpo_gym import *
from utilit.fit_pdf import FitPDF
from utilit.stop_cond import stop_vol

def main(args):
    model_name = args.model_name

    b = BenchmarkSet(scenario=model_name)
    data_name_all = b.instances
    for data_id in range(args.data_id):
        args.data_name= data_name_all[data_id]
        api1 = P_NN(task_id=args.data_name, alg_name=args.alg_name)
        n_int,cost_hf_all =5, []
        for i in range(n_int):
            cs=get_hyperparameter_search_space()
            config = cs.sample_configuration()
            f, cost = api1.fit(config, budget=api1.bound_fid[1])
            cost_hf=cost
            cost_hf_all.append(cost_hf)
        cost_hf_m=sum(cost_hf_all)/n_int
        args.cost_hf_m = cost_hf_m
        n_iters_all = [api1.d, 20 - api1.d]
        n_cost_all = [n_iters_all[0] * args.cost_hf_m, n_iters_all[1] * args.cost_hf_m]
        args.cost_hf_m = cost_hf_m
        for seed in range(args.max_run):
            args.seed=seed
            if 'prior' in args.alg_name:
                args.budget = [4,100]
                args.maxiter=[5*api1.d,20- api1.d]
            api = P_NN(task_id=args.data_name,alg_name=args.alg_name)
            random_search(args, api)

def random_search(args, api):
    random.seed(args.seed)
    np.random.seed(args.seed)
    cs =get_hyperparameter_search_space()
    stop_vol_cal = stop_vol(cs=cs)
    kde_vartypes, vartypes = get_vartypes(cs)
    Ymin=[]
    rt = 0
    inc_valid = np.inf
    max_budget = args.budget[-1]
    runtime = [0]
    bd_prom,budget=[],[]
    maxiter=90

    n_iters_all = [api.d, 20 - api.d]
    n_cost_all = [n_iters_all[0] * args.cost_hf_m, n_iters_all[1] * args.cost_hf_m]
    it=0
    stop_cond =False
    j=0
    while runtime[-1] < n_cost_all[j] and stop_cond == False:
        if it==0:
            runtime=[]
        key = cs.sample_configuration()
        loss, cost = api.fit(config=key, budget=args.budget[j])
        if args.budget[j] == max_budget:
            if inc_valid > api.f[-1]:
                inc_valid = api.f[-1]
        Ymin.append(float(inc_valid))
        rt += api.cost_all[-1]
        runtime.append(float(rt))
        it+=1
        Samp, YS = np.array(api.X), np.array(api.f)
        if Samp.shape[0] > 10:
            stop_cond=stop_vol_cal(Samp,YS.reshape(-1,1))

    Samp, YS = np.array(api.X), np.array(api.f)
    ind = np.argsort(YS)[::1]
    n = int(len(YS) / 2)
    X_prom = Samp[ind[:n], :]
    j=1
    while runtime[-1] < n_cost_all[j] :
        train_data_good = np.array(X_prom)  # 2D 数据
        pdf_fitter = FitPDF(cs, train_data_good, kde_vartypes)
        pdf_fitter.fit_pdf_pro()
        weighted_samples = pdf_fitter.sample_weighted(data_min=np.array(api.bound[:,0]),
                                                      data_max=np.array(api.bound[:,1]),
                                                      n_samples=1000, alpha=0.5)
        candi = np.array(weighted_samples)
        random_row = random.choice(candi)
        s = cs.sample_configuration()
        name = s.keys()
        for j1 in range(len(cs)):
            if j1 in api.d_int:
                s[name[j1]] = int(random_row[ j1])
            else:
                s[name[j1]] = random_row[ j1]
        key = s
        loss, cost = api.fit(config=key, budget=args.budget[j])
        if args.budget[j]==max_budget:
            if inc_valid > api.f[-1]:
                inc_valid = api.f[-1]
        Ymin.append(float(inc_valid))
        rt += api.cost_all[-1]
        runtime.append(float(rt))

    Data_xg = []
    Data_xg.append({
        "seed": args.seed,
        "X_value": api.X,
        "fidelity": api.fidelity,
        'f': api.f,
        'cost': api.cost_all,
        'Ymin': Ymin,
        'runtime': runtime,'bd_prom': bd_prom
    })


    output_path = './Result/'  + args.alg_name + '/' + args.fidelity_name+ '/'
    if os.path.exists(output_path) == False:
        os.makedirs(output_path, exist_ok=True)
    file_name = args.model_name + str(args.data_name) +'_x_fid_'+str(args.x_fid)+ str(budget) + 'r' + str(args.seed) + ".pkl"

    file_rout = os.path.join(output_path, file_name)
    pickle.dump(Data_xg, open(file_rout, 'wb'))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alg_name", "-a", dest="alg_name", type=str, default='random_search_prior',choices=['random_search','random_search_prior'])
    args.add_argument("--task", "-t", dest="task", type=int, default=4)
    args.add_argument("--data_id", "-d", dest="data_id", type=int, default=34)#newsgroups covtype      '','lfw_people','covtype','mnist_784'
    args.add_argument("--model_name", "-m", dest="model_name", type=str, default='fcnet',choices=['lcbench','fcnet','pd1','jahs','nb301','nb201'])
    args.add_argument("--fidelity_name", "-f", dest="fidelity_name", type=str, default='epoch')#data_size
    args.add_argument("--max_run", "-r", dest="max_run", type=int, default=11)
    args.add_argument("--x_fid", "-x", dest="x_fid", type=float, default=1)
    args = args.parse_args()
    main(args)

