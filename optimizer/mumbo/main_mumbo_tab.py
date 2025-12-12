import os,pickle
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"
import argparse
import numpy as np
from optimizer.sgp.SGP import MF_Opt
from optimizer.mumbo.MUMBO import MF_Opt as MUMBO_Opt
from optimizer.utilit.Initialization_mix import  initial
from Benchmark.P_lcbbench import   P_lcb
from Benchmark.P_fcnet import   P_NN
from Benchmark.P_pd1  import P_pd1
from Benchmark.P_jahs  import P_jahs
from Benchmark.P_nb301  import P_nb301
from Benchmark.P_nb201  import P_nb201

from yahpo_gym import *


def main(args):

    if args.model_name=="lcbench":
        b = BenchmarkSet(scenario=args.model_name)
        bound_fid=[1,50]
        data_name_all = b.instances
        for data_id in range(args.data_id):
            args.data_name= data_name_all[data_id]
            for seed in range(args.max_run):
                args.seed=seed
                if 'prior' in args.alg_name:
                    args.budget = [10]
                    args.maxN=50
                    api = P_lcb(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                    Data_xg = sgp(args, api)
                    args.budget = [10,bound_fid[-1]]
                    args.maxN=70
                    api = P_lcb(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                    Data_xg = mumbo(args, api, prior=Data_xg)
                else:
                    args.maxN=90
                    args.budget = [10,bound_fid[-1]]
                    api = P_lcb( task_id=args.data_name,alg_name=args.alg_name,x_fid=args.budget)
                    Data_xg=mumbo(args, api)

    elif args.model_name=="fcnet":
        b = BenchmarkSet(scenario=args.model_name)
        bound_fid=[1,100]
        data_name_all = b.instances
        for data_id in [args.data_id]:
            args.budget = [bound_fid[-1]]
            args.data_name= data_name_all[data_id]
            for seed in [args.max_run]:
                args.seed=seed
                if 'prior' in args.alg_name:
                    for budget_lf in [4,10, 20 ,30 ,40, 60, 80]:
                        args.budget = [budget_lf]
                        args.maxN=30
                        api = P_NN(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = sgp(args, api)
                        args.budget = [4,bound_fid[-1]]
                        args.budget_all = [budget_lf,bound_fid[-1]]

                        args.maxN=70
                        api = P_NN(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = mumbo(args, api, prior=Data_xg)
                else:
                    for budget_lf in [4]:
                        args.maxN=90
                        args.budget = [budget_lf,bound_fid[-1]]
                        api = P_NN( task_id=args.data_name,alg_name=args.alg_name,x_fid=args.budget)
                        Data_xg=mumbo(args, api)

    elif args.model_name=="pd1":
        data_name_all = ['lm1b_transformer_2048', 'translatewmt_xformer_64',
                         'cifar100_wideresnet_2048', 'imagenet_resnet_512']
        for data_id in [args.data_id]:
            args.data_name = data_name_all[data_id]
            api = P_pd1(task_id=args.data_name, alg_name=args.alg_name)
            bound_fid = api.bound_fid
            if args.data_name == 'lm1b_transformer_2048':
                budget_lf_all = [4, 10, 20, 30, 40, 60]
            elif args.data_name == 'translatewmt_xformer_64':
                budget_lf_all = [4, 8, 12]
            elif args.data_name == 'cifar100_wideresnet_2048':
                budget_lf_all = [45, 60, 80, 120, 150]
            elif args.data_name == 'imagenet_resnet_512':
                budget_lf_all = [4, 10, 20, 30, 40, 60, 80]
            for seed in [args.max_run]:
                args.seed = seed
                if 'prior' in args.alg_name:
                    for budget_lf in budget_lf_all:
                        args.budget = [budget_lf]
                        args.maxN =30
                        api = P_pd1(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = sgp(args, api)
                        args.budget = [budget_lf_all[0], bound_fid[-1]]
                        args.budget_all = [budget_lf,bound_fid[-1]]

                        args.maxN = 70
                        api = P_pd1(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = mumbo(args, api, prior=Data_xg)
                else:
                    for budget_lf in [budget_lf_all[0]]:
                        args.maxN = 90
                        args.budget = [budget_lf, bound_fid[-1]]
                        api = P_pd1(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = mumbo(args, api)


    elif args.model_name=="jahs":
        data_name_all = ['CIFAR10', 'ColorectalHistology', 'FashionMNIST']
        for data_id in [args.data_id]:
            args.data_name = data_name_all[data_id]
            api = P_jahs(task_id=args.data_name, alg_name=args.alg_name)
            bound_fid = api.bound_fid
            for seed in [args.max_run]:
                args.seed = seed
                if 'prior' in args.alg_name:
                    for budget_lf in [4,10 ,20 ,30 ,40 ,60 ,80, 120, 150]:
                        args.budget = [budget_lf]
                        args.maxN = 50
                        api = P_jahs(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = sgp(args, api)
                        args.budget = [4, bound_fid[-1]]
                        args.budget_all = [budget_lf,bound_fid[-1]]

                        args.maxN = 70
                        api = P_jahs(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = mumbo(args, api, prior=Data_xg)
                else:
                    for budget_lf in [4]:
                        args.maxN = 90
                        args.budget = [budget_lf, bound_fid[-1]]
                        api = P_jahs(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = mumbo(args, api)


    elif args.model_name=="nb301":
        b = BenchmarkSet(scenario=args.model_name)
        bound_fid=[1,98]
        data_name_all = b.instances
        for data_id in [args.data_id]:
            args.data_name = data_name_all[data_id]
            api = P_nb301(task_id=args.data_name, alg_name=args.alg_name)
            bound_fid = api.bound_fid
            for seed in [args.max_run]:
                args.seed = seed
                if 'prior' in args.alg_name:
                    for budget_lf in [4, 10, 20, 30, 40, 60, 80]:
                        args.budget = [budget_lf]
                        args.maxN = 50
                        api = P_nb301(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = sgp(args, api)
                        args.budget = [4, bound_fid[-1]]
                        args.budget_all = [budget_lf,bound_fid[-1]]

                        args.maxN = 70
                        api = P_nb301(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = mumbo(args, api, prior=Data_xg)
                else:
                    for budget_lf in [4]:
                        args.maxN = 90
                        args.budget = [budget_lf, bound_fid[-1]]
                        api = P_nb301(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = mumbo(args, api)
    elif args.model_name=="nb201":
        data_name_all = ['cifar10', 'cifar100', 'imagenet']
        for data_id in [args.data_id]:
            args.data_name = data_name_all[data_id]
            api = P_nb201(task_id=args.data_name, alg_name=args.alg_name)
            bound_fid = api.bound_fid
            for seed in [args.max_run]:
                args.seed = seed
                if 'prior' in args.alg_name:
                    for budget_lf in [4,10 ,20 ,30 ,40 ,60 ,80 ,120 ,150]:
                        args.budget = [budget_lf]
                        args.maxN = 30
                        api = P_nb201(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = sgp(args, api)
                        args.budget = [4, bound_fid[-1]]
                        args.budget_all = [budget_lf,bound_fid[-1]]

                        args.maxN = 70
                        api = P_nb201(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = mumbo(args, api, prior=Data_xg)
                else:
                    for budget_lf in [4]:
                        args.maxN = 90
                        args.budget = [budget_lf, bound_fid[-1]]
                        api = P_nb201(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                        Data_xg = mumbo(args, api)

def sgp(args, api,prior=[],rt=0):
    model_M, FM_name,AC= 'SGP','SGP', 'LCB'
    dim=api.d
    N0, maxN=  2 * dim,args.maxN
    N_train = np.array([N0])
    Param=initial(N_train=N_train,maxN=maxN,AC=AC,model_M=model_M,FM_name = FM_name,r=args.seed)
    Param.Kint,Param.NorY,Param.prior,Param.rt='Mat','log',prior,rt
    op1 = MF_Opt(fo=api,Param=Param)
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
    Param=initial(N_train=N_train,maxN=maxN,AC=AC,model_M=model_M,FM_name = FM_name,r=args.seed)
    Param.Kint,Param.NorY,Param.prior,Param.rt='Mat','log',prior,rt
    op1 = MUMBO_Opt(fo=api,Param=Param)
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
    args.add_argument("--alg_name", "-a", dest="alg_name", type=str, default='mumbo',choices=['mumbo','mumbo_prior'])
    args.add_argument("--model_name", "-m", dest="model_name", type=str, default='lcbench',choices=['lcbench','fcnet','pd1','jahs','nb301','nb201'])
    args.add_argument("--seed", "-s", dest="seed", type=int, default=0)
    args.add_argument("--data_id", "-d", dest="data_id", type=int, default=2)#newsgroups covtype      '','lfw_people','covtype','mnist_784'
    args.add_argument("--fidelity_name", "-n", dest="fidelity_name", type=str, default='epoch')#data_size
    args.add_argument("--max_run", "-r", dest="max_run", type=int, default=1)
    args.add_argument("--x_fid", "-x", dest="x_fid", type=float, default=1)
    args = args.parse_args()
    main(args)






