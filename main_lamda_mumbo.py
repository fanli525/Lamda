import os,pickle
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"
import argparse
import numpy as np
from optimizer.sgp.SGP import MF_Opt
from optimizer.mumbo.MUMBO import MF_Opt as MUMBO_Opt
from optimizer.utilit.Initialization_mix import  initial
from P_fcnet  import P_NN,get_hyperparameter_search_space,get_vartypes

from yahpo_gym import *


def main(args):

    b = BenchmarkSet(scenario=args.model_name)
    bound_fid=[1,100]
    data_name_all = b.instances
    for data_id in [0,1,2,3]:
        args.budget = [bound_fid[-1]]
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
        n_iters_all = [1,2]
        n_cost_all = [n_iters_all[0] * args.cost_hf_m, n_iters_all[1] * args.cost_hf_m]

        for seed in [args.max_run]:
            args.seed=seed
            if 'prior' in args.alg_name:
                for budget_lf in [4]:
                    args.budget = [budget_lf]
                    args.maxN=n_cost_all[0]
                    api = P_NN(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                    Data_xg = sgp(args, api)
                    args.budget = [4,bound_fid[-1]]
                    args.budget_all = [budget_lf,bound_fid[-1]]

                    args.maxN=n_cost_all[1]
                    api = P_NN(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                    Data_xg = mumbo(args, api, prior=Data_xg)
            else:
                for budget_lf in [4]:
                    n_iters = 20
                    n_cost_all = n_iters * args.cost_hf_m
                    args.maxN = n_cost_all
                    args.budget = [budget_lf,bound_fid[-1]]
                    api = P_NN( task_id=args.data_name,alg_name=args.alg_name,x_fid=args.budget)
                    Data_xg=mumbo(args, api)

def sgp(args, api,prior=[],rt=0):
    model_M, FM_name,AC= 'SGP','SGP', 'LCB'
    dim=api.d
    N0, maxN=  2 * dim,args.maxN
    N_train = np.array([N0])
    Param=initial(N_train=N_train,maxcost=maxN,AC=AC,model_M=model_M,FM_name = FM_name,r=args.seed)
    Param.Kint,Param.NorY,Param.prior,Param.rt='Mat','log',prior,rt
    if args.model_name in ['mfh3','mfh6']:
        Param.NorY=''

    cs=get_hyperparameter_search_space()
    kde_vartypes, vartypes = get_vartypes(cs)
    op1 = MF_Opt(fo=api,Param=Param,cs=cs,kde_vartypes=kde_vartypes,vartypes=vartypes)
    Data_xg = op1()

    output_path = './Result/' + args.alg_name + '/' + args.fidelity_name + '/'
    if os.path.exists(output_path) == False:
        os.makedirs(output_path, exist_ok=True)
    file_name = args.model_name + str(args.data_name) +'_x_fid_'+str(args.x_fid)+ str(args.budget)+ 'r' + str(args.seed) + ".pkl"
    file_rout = os.path.join(output_path, file_name)
    pickle.dump(Data_xg, open(file_rout, 'wb'))
    return Data_xg

def mumbo(args, api,prior=[],rt=0):
    model_M, FM_name,AC= 'ICM','ICM', 'LCB'
    dim=api.d
    if len(prior)==0:
        N0, maxN=   dim+6 ,args.maxN
    else:
        N0, maxN=  min(dim,10),args.maxN
    N_train = np.array([N0])
    Param=initial(N_train=N_train,maxcost=maxN,AC=AC,model_M=model_M,FM_name = FM_name,r=args.seed)
    Param.Kint,Param.NorY,Param.prior,Param.rt='Mat','log',prior,rt
    if args.model_name in ['mfh3','mfh6']:
        Param.NorY=''
    cs=get_hyperparameter_search_space()
    kde_vartypes, vartypes = get_vartypes(cs)
    op1 = MUMBO_Opt(fo=api,Param=Param,cs=cs,kde_vartypes=kde_vartypes,vartypes=vartypes)

    Data_xg = op1()

    output_path = './Result/' + args.alg_name + '/' + args.fidelity_name + '/'
    if os.path.exists(output_path) == False:
        os.makedirs(output_path, exist_ok=True)
    file_name = args.model_name + str(args.data_name) +'_x_fid_'+str(args.x_fid)+ str(args.budget_all)+ 'r' + str(args.seed) + ".pkl"
    file_rout = os.path.join(output_path, file_name)
    pickle.dump(Data_xg, open(file_rout, 'wb'))
    return Data_xg




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alg_name", "-a", dest="alg_name", type=str, default='mumbo_prior',choices=['mumbo','mumbo_prior'])
    args.add_argument("--model_name", "-m", dest="model_name", type=str, default='fcnet')
    args.add_argument("--seed", "-s", dest="seed", type=int, default=0)
    args.add_argument("--data_id", "-d", dest="data_id", type=int, default=0)#newsgroups covtype      '','lfw_people','covtype','mnist_784'
    args.add_argument("--fidelity_name", "-n", dest="fidelity_name", type=str, default='data_size')#data_size
    args.add_argument("--max_run", "-r", dest="max_run", type=int, default=11)
    args.add_argument("--x_fid", "-x", dest="x_fid", type=float, default=1)
    args = args.parse_args()
    main(args)






