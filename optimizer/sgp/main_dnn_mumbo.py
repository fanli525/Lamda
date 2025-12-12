import os,pickle
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"
import argparse
import numpy as np
from optimizer.sgp.SGP import MF_Opt
from optimizer.sgp.MUMBO import MF_Opt as MUMBO_Opt
from optimizer.utilit.Initialization_mix import  initial
from Benchmark.P_resnet_gp import NN



def main(args):
    if args.fidelity_name == 'epoch':
        bound_fid = [1, 200]
    else:
        bound_fid = [0.3, 1]
    if 'prior' in args.alg_name:
        if args.fidelity_name=='epoch':
            args.budget = [20]
        else:
            args.budget = [0.3]
        args.maxN=10
        api = NN(alg_name=args.alg_name, seed=args.seed, data_name=args.data_name, data_type=args.data_type,
                 model_name=args.model_name,
                 fidelity_name=args.fidelity_name, x_fid1=args.x_fid, x_fid=args.budget, sch_name=args.sch_name, source=args.source)

        Data_xg = sgp(args, api)
        if args.fidelity_name == 'epoch':
            args.budget = [20,bound_fid[-1]]
        else:
            args.budget = [0.3,bound_fid[-1]]
        args.maxN=10
        api = NN(alg_name=args.alg_name, seed=args.seed, data_name=args.data_name, data_type=args.data_type,
                 model_name=args.model_name,
                 fidelity_name=args.fidelity_name, x_fid1=args.x_fid, x_fid=args.budget, sch_name=args.sch_name, source=args.source)
        Data_xg = mumbo(args, api, bd_prom=Data_xg[0]['bd_prom'])
    else:
        args.maxN=10
        if args.fidelity_name == 'epoch':
            args.budget = [20,bound_fid[-1]]
        else:
            args.budget = [0.3,bound_fid[-1]]
        api = NN(alg_name=args.alg_name, seed=args.seed, data_name=args.data_name, data_type=args.data_type,
                 model_name=args.model_name,
                 fidelity_name=args.fidelity_name, x_fid1=args.x_fid, x_fid=args.budget, sch_name=args.sch_name, source=args.source)
        Data_xg=mumbo(args, api)


def sgp(args, api,bd_prom=[]):
    model_M, FM_name,AC= 'SGP','SGP', 'LCB'
    dim=api.d
    N0, maxN=  1* dim,args.maxN
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

def mumbo(args, api,bd_prom=[]):
    model_M, FM_name,AC= 'ICM','ICM', 'LCB'
    dim=api.d
    N0, maxN=  1 * dim,args.maxN
    N_train = np.array([N0])
    Param=initial(N_train=N_train,maxN=maxN,AC=AC,model_M=model_M,FM_name = FM_name,r=args.seed)
    Param.Kint,Param.NorY,Param.bd_prom='Mat','log',bd_prom
    op1 = MUMBO_Opt(fo=api,Param=Param)
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
    args.add_argument("--alg_name", "-a", dest="alg_name", type=str, default='mumbo',choices=['mumbo','mumbo_prior'])
    args.add_argument("--fun", "-f", dest="fun", type=int, default=0)
    args.add_argument("--seed", "-s", dest="seed", type=int, default=0)
    args.add_argument("--data_name", "-d", dest="data_name", type=str, default='cifar10')
    args.add_argument("--model_name", "-m", dest="model_name", type=str, default='lenet', choices=['lenet','alexnet','resnet18','resnet50','resnet101','vgg16','vgg19'])
    args.add_argument("--sch_name", "-c", dest="sch_name", type=str, default='CosineAnnealingLR1')
    args.add_argument("--source", "-o", dest="source", type=str, default='')
    args.add_argument("--fidelity_name", "-n", dest="fidelity_name", type=str, default='datasize', choices=['epoch', 'datasize'])
    args.add_argument("--x_fid", "-x", dest="x_fid", type=float, default=0.001)
    args.add_argument("--data_type", "-e", dest="data_type", type=int, default=32)
    args = args.parse_args()
    main(args)





