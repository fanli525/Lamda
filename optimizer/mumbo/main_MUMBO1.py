import os,pickle
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"
import argparse
import numpy as np
from SGP import MF_Opt
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
                    args.budget = [50]
                    args.maxN=40
                    api = P_lcb(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                    Data_xg = sgp(args, api)
                    args.budget = [bound_fid[-1]]
                    args.maxN=70
                    api = P_lcb(task_id=args.data_name, alg_name=args.alg_name, x_fid=args.budget)
                    Data_xg = sgp(args, api, bd_prom=Data_xg[0]['bd_prom'])
                else:
                    args.maxN=90
                    args.budget = [bound_fid[-1]]
                    api = P_lcb( task_id=args.data_name,alg_name=args.alg_name,x_fid=args.budget)
                    Data_xg=sgp(args, api)

    elif args.model_name=="fcnet":
        b = BenchmarkSet(scenario=args.model_name)
        bound_fid=[1,100]
        data_name_all = b.instances
        for data_id in range(args.data_id):
            args.budget = [bound_fid[-1]]
            args.data_name= data_name_all[data_id]
            for seed in range(args.max_run):
                args.seed=seed
                api = P_NN( task_id=args.data_name,alg_name=args.alg_name,x_fid=args.budget)
                sgp(args, api)
    elif args.model_name=="pd1":
        data_name_all = ['lm1b_transformer_2048', 'translatewmt_xformer_64', 'uniref50_transformer_128',
                         'cifar100_wideresnet_2048', 'imagenet_resnet_512']
        for data_id in range(args.data_id):
            args.data_name = data_name_all[data_id]
            api = P_pd1(task_id=args.data_name, alg_name=args.alg_name)
            bound_fid = api.bound_fid
            args.budget = [bound_fid[-1]]

            for seed in range(args.max_run):
                args.seed=seed
                api = P_pd1( task_id=args.data_name,alg_name=args.alg_name,x_fid=args.budget)
                sgp(args, api)
    elif args.model_name=="jahs":
        data_name_all = ['CIFAR10', 'ColorectalHistology', 'FashionMNIST']
        for data_id in range(args.data_id):
            args.data_name = data_name_all[data_id]
            api = P_jahs(task_id=args.data_name, alg_name=args.alg_name)
            bound_fid = api.bound_fid
            args.budget = [bound_fid[-1]]
            for seed in range(args.max_run):
                args.seed=seed
                api = P_jahs( task_id=args.data_name,alg_name=args.alg_name,x_fid=args.budget)
                sgp(args, api)
    elif args.model_name=="nb301":
        b = BenchmarkSet(scenario=args.model_name)
        bound_fid=[1,98]
        data_name_all = b.instances
        for data_id in range(args.data_id):
            args.budget = [bound_fid[-1]]
            args.data_name= data_name_all[data_id]
            for seed in range(args.max_run):
                args.seed=seed
                api = P_nb301( task_id=args.data_name,alg_name=args.alg_name,x_fid=args.budget)
                sgp(args, api)
    elif args.model_name=="nb201":
        data_name_all = ['cifar10', 'cifar100', 'imagenet']
        for data_id in range(args.data_id):
            args.data_name = data_name_all[data_id]
            api = P_nb201(task_id=args.data_name, alg_name=args.alg_name)
            bound_fid = api.bound_fid
            args.budget = [bound_fid[-1]]
            for seed in range(args.max_run):
                args.seed = seed
                api = P_nb201( task_id=args.data_name,alg_name=args.alg_name,x_fid=args.budget)
                sgp(args, api)


def sgp(args, api,bd_prom=[]):
    model_M, FM_name,AC= 'SGP','SGP', 'LCB'
    dim=api.d
    N0, maxN=  2 * dim,args.maxN
    N_train = np.array([N0])
    Param=initial(N_train=N_train,maxN=maxN,AC=AC,model_M=model_M,FM_name = FM_name,r=args.seed)
    Param.Kint,Param.NorY,Param.bd_prom='Mat','log',bd_prom
    op1 = MF_Opt(fo=api,Param=Param)
    Data_xg = op1()

    output_path = './Result/' + args.alg_name + '/' + args.fidelity_name + '/'
    if os.path.exists(output_path) == False:
        os.makedirs(output_path, exist_ok=True)
    file_name = args.model_name + str(args.data_name) +'_x_fid_'+str(args.x_fid)+ str(args.budget)+ 'r' + str(args.seed) + ".pkl"
    file_rout = os.path.join(output_path, file_name)
    pickle.dump(Data_xg, open(file_rout, 'wb'))
    return Data_xg


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alg_name", "-a", dest="alg_name", type=str, default='sgp_prior',choices=['mumbo','sgp'])
    args.add_argument("--fun", "-f", dest="fun", type=int, default=0)
    args.add_argument("--model_name", "-m", dest="model_name", type=str, default='lcbench',choices=['lcbench','fcnet','pd1','jahs','nb301','nb201'])
    args.add_argument("--seed", "-s", dest="seed", type=int, default=0)
    args.add_argument("--data_id", "-d", dest="data_id", type=int, default=1)#newsgroups covtype      '','lfw_people','covtype','mnist_784'
    args.add_argument("--fidelity_name", "-n", dest="fidelity_name", type=str, default='epoch')#data_size
    args.add_argument("--max_run", "-r", dest="max_run", type=int, default=1)
    args.add_argument("--x_fid", "-x", dest="x_fid", type=float, default=1)
    args.add_argument("--budget", "-b", dest="budget", type=str, default=[1])

    args = args.parse_args()
    main(args)





