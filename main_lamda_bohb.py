import logging
logging.basicConfig(level=logging.ERROR)
from hpbandster.core.worker import Worker
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
import numpy as np
import argparse
import os, pickle
from P_fcnet  import P_NN,get_hyperparameter_search_space
import random
from yahpo_gym import *
def main(args):
    model_name = args.model_name
    b = BenchmarkSet(scenario=model_name)
    data_name_all = b.instances
    for data_id in [0,1,2,3]:
        args.data_name= data_name_all[data_id]
        api1 = P_NN(task_id=args.data_name, alg_name=args.alg_name)
        n_int,cost_hf_all =5, []
        for i in range(n_int):
            cs=get_hyperparameter_search_space()
            config = cs.sample_configuration()
            res = api1.fit(config, budget=api1.bound_fid[1])
            cost_hf=res['info']
            cost_hf_all.append(cost_hf)
        cost_hf_m=sum(cost_hf_all)/n_int
        args.cost_hf_m = cost_hf_m
        for seed in range(args.max_run):
            args.seed=seed
            for budget_lf in [4]:
                args.budget_lf =budget_lf
                api = P_NN(task_id=args.data_name, alg_name= args.alg_name)
                bohb(args, api)


class MyWorker(Worker):
    def __init__(
            self,
            *args,
            api=None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.api = api
    def compute(self, config, budget, *args, **kwargs):
        res = self.api.fit(config, budget)
        return res



def bohb(args, api):
    random.seed(args.seed)
    np.random.seed(args.seed)
    cs =get_hyperparameter_search_space()


    min_budget, max_budget= api.bound_fid[0], api.bound_fid[1]
    min_bandwidth = .3
    num_samples = 64
    bandwidth_factor = 3
    n_iters_all=[api.d,20- api.d]
    n_cost_all = [n_iters_all[0]*args.cost_hf_m,n_iters_all[1]*args.cost_hf_m]
    hb_run_id = str(args.seed)
    NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=7891)
    ns_host, ns_port = NS.start()
    num_workers = 1
    workers = []
    for i in range(num_workers):
        w = MyWorker(api=api, nameserver=ns_host, nameserver_port=ns_port,
                     run_id=hb_run_id,
                     id=i)
        w.run(background=True)
        workers.append(w)
    # budget = []
    N_all, it = [], 0
    prior ,prior_opt= [],[]
    budget1=[0,args.budget_lf,api.bound_fid[1]]
    for bg in [[args.budget_lf],[4,api.bound_fid[1]]]:
        n_iters = n_iters_all[it]
        max_budget = bg[-1]
        budget = bg
        if it== 1:
            HF=True
            random_fraction = .1
            eta =3
            min_budget,max_budget = api.bound_fid[0],api.bound_fid[1]
            budget =[]
            model='all'
        else:
            HF=False
            random_fraction = .33
            eta = 1 + 1.0e-6
            model='single'

        bohb = BOHB(configspace=cs,
                    run_id=hb_run_id,
                    eta=eta, min_budget=min_budget, max_budget=max_budget,
                    nameserver=ns_host,
                    nameserver_port=ns_port,
                    num_samples=num_samples,
                    random_fraction=random_fraction, bandwidth_factor=bandwidth_factor,
                    ping_interval=10, min_bandwidth=min_bandwidth, budget=budget,prior=prior,model=model)
        results = bohb.run(n_iterations=n_iters, min_n_workers=num_workers,alg_name=args.alg_name,HF=HF,maxcost=n_cost_all[it])
        if it==0:
            X_lf = np.array(api.X)
            d_int =  api.d_int
            f_lf = np.array( api.f)
            ind = np.argsort(f_lf)[::1]
            n = min(int(len(f_lf) / 3), 10 * (X_lf.shape[1]))
            prior_x1 = X_lf[ind[:n],:]
            prior_x = []
            for i in range(prior_x1.shape[0]):
                s=cs.sample_configuration()
                name=s.keys()
                for j in range(len(cs)):
                    if j in d_int:
                        s[name[j]]=int(prior_x1[i,j])
                    else:
                        s[name[j]]=prior_x1[i,j]
                prior_x.append(s)
            prior_y=f_lf[ind[:n]]
            prior.append({
                "prior_x":prior_x, "loss": prior_y,
                "budget":api.bound_fid[1]
            })


        it=it+1

    Ymin,rt ,inc_valid,runtime=[],0,np.inf,[]
    for i in range(len(api.f)):
        if api.fidelity[i] == max_budget:
            if inc_valid>api.f[i]:
                inc_valid = api.f[i]
        Ymin.append(float(inc_valid))
        rt += api.cost_all[i]
        runtime.append(float(rt))
    print(f"{args.model_name}{args.data_name}r{args.seed} opt: {Ymin[-1]} cost: {np.round(runtime[-1], 3)}")

    Data_xg = []
    Data_xg.append({
        "seed": args.seed,
        "X_value": api.X,
        "fidelity": api.fidelity,
        'f': api.f,
        'cost': api.cost_all,
        'Ymin': Ymin,
        'runtime': runtime
    })


    output_path = './Result/'  + args.alg_name + '/' + args.fidelity_name+ '/'
    if os.path.exists(output_path) == False:
        os.makedirs(output_path, exist_ok=True)
    file_name = args.model_name + str(args.data_name) +'_x_fid_'+str(args.x_fid)+ str(budget1) + 'r' + str(args.seed) + ".pkl"

    file_rout = os.path.join(output_path, file_name)
    pickle.dump(Data_xg, open(file_rout, 'wb'))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alg_name", "-a", dest="alg_name", type=str, default='bohb_prior_adp')
    args.add_argument("--task", "-t", dest="task", type=int, default=4)
    args.add_argument("--data_id", "-d", dest="data_id", type=int, default=0)#
    args.add_argument("--model_name", "-m", dest="model_name", type=str, default='fcnet')
    args.add_argument("--fidelity_name", "-f", dest="fidelity_name", type=str, default='epoch')#data_size
    args.add_argument("--max_run", "-r", dest="max_run", type=int, default=2)
    args.add_argument("--x_fid", "-x", dest="x_fid", type=float, default=1)
    args = args.parse_args()
    main(args)

