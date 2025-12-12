import matplotlib.pyplot as plt
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
    for args.data_name in data_name_all:
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
            api = P_NN(task_id=args.data_name)
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

    eta = 3
    n_iters = 20
    n_cost_all =n_iters*args.cost_hf_m

    min_budget ,max_budget= api.bound_fid[0],api.bound_fid[1]
    min_bandwidth = .3
    num_samples = 64
    random_fraction = .33
    bandwidth_factor = 3

    budget = []
    hb_run_id = str(args.seed)
    NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=8990)
    ns_host, ns_port = NS.start()
    num_workers = 1
    workers = []
    for i in range(num_workers):
        w = MyWorker(api=api,nameserver=ns_host, nameserver_port=ns_port,
                     run_id=hb_run_id,
                     id=i)
        w.run(background=True)
        workers.append(w)

    bohb = BOHB(configspace=cs,
                run_id=hb_run_id,
                eta=eta, min_budget=min_budget, max_budget=max_budget,
                nameserver=ns_host,
                nameserver_port=ns_port,
                num_samples=num_samples,
                random_fraction=random_fraction, bandwidth_factor=bandwidth_factor,
                ping_interval=10, min_bandwidth=min_bandwidth)
    results = bohb.run(n_iterations=n_iters, min_n_workers=num_workers,alg_name=args.alg_name,maxcost=n_cost_all)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    Ymin,runtime,rt,inc_valid=[],[],0,np.inf
    for i in range(len(api.f)):
        if api.fidelity[i] == max_budget:
            if inc_valid>api.f[i]:
                inc_valid = api.f[i]
        Ymin.append(float(inc_valid))
        rt += api.cost_all[i]
        runtime.append(float(rt))


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

    print(f"{args.model_name}{args.data_name}r{args.seed} opt: {Ymin[-1]} cost: {np.round(runtime[-1], 3)}")

    output_path = './Result/'  + args.alg_name + '/' + args.fidelity_name+ '/'
    if os.path.exists(output_path) == False:
        os.makedirs(output_path, exist_ok=True)
    file_name = args.model_name + str(args.data_name) +'_x_fid_'+str(args.x_fid)+ str(budget) + 'r' + str(args.seed) + ".pkl"

    file_rout = os.path.join(output_path, file_name)
    pickle.dump(Data_xg, open(file_rout, 'wb'))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alg_name", "-a", dest="alg_name", type=str, default='bohb')
    args.add_argument("--task", "-t", dest="task", type=int, default=4)
    args.add_argument("--data_id", "-d", dest="data_id", type=int, default=34)
    args.add_argument("--model_name", "-m", dest="model_name", type=str, default='fcnet')
    args.add_argument("--fidelity_name", "-f", dest="fidelity_name", type=str, default='epoch')#data_size
    args.add_argument("--max_run", "-r", dest="max_run", type=int, default=2)
    args.add_argument("--x_fid", "-x", dest="x_fid", type=float, default=1)
    args = args.parse_args()
    main(args)

