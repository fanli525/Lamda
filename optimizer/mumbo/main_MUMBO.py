import os
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"
import argparse
import numpy as np
from optimizer.mumbo.MUMBO import MF_Opt
from optimizer.utilit.Initialization_mix import  initial
from Benchmark.P_lcbbench import   P_lcb
from yahpo_gym import *

def main(args):

    maxN=20
    AC='LCB'
    model_M='ICM'
    model_name="lcbench"
    b = BenchmarkSet(scenario=model_name)
    taskname=b.instances
    output_path = model_name + '/'
    ind_task=args.fun
    task_id= taskname[ind_task]

    if model_name=="lcbench":
        bound_fid=[1,50]

        x_fid = [10, bound_fid[-1]]

        fo =P_lcb( task_id=task_id,alg_name=args.alg_name,x_fid=x_fid)
    else:
        bound_fid=[1,100]
        x_fid = [5, 25, bound_fid[-1]]
        fo =P_NN( task_id=task_id,task=len(x_fid),x_fid=x_fid)

    dim=fo.d
    N1 =1* dim
    N_train = np.array(len(x_fid) * [N1])
    r=args.seed
    Param=initial(N_train=N_train,maxN=maxN,AC=AC,model_M=model_M,r=r)
    Param.Kint,Param.NorY='Mat','log'
    op1 = MF_Opt(fo=fo,Param=Param)

    Data_xg = op1()
    # from util.save_data import Save
    # output_path = './' + model_M + '/'
    # Save = Save(output_path)  #
    # Save.save_incumbent(Data_xg,'mumbo_' +Param.Kint+model_M + '_' +  task_id + '_x_fid_' + str(x_fid) + '_maxN_' + str(maxN)+ '_r_' + str(r))
    #



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alg_name", "-a", dest="alg_name", type=str, default='mumbo')
    args.add_argument("--fun", "-f", dest="fun", type=int, default=0)
    args.add_argument("--model_name", "-m", dest="model_name", type=str, default='lcbench')
    args.add_argument("--seed", "-s", dest="seed", type=int, default=0)
    args = args.parse_args()
    main(args)












