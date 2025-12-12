
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.examples.fabolas import fmin_fabolas

import os
import json
import numpy as np
from Benchmark.PM import  P_NN,rnet,P_lcb
from yahpo_gym import *
from yahpo_gym import local_config
local_config.init_config()
local_config.set_data_path("./yahpo_data-main/")

maxN=20
model_M='ICM'
model_name="lcbench"
b = BenchmarkSet(scenario=model_name)
taskname=b.instances

fid_ind = 2
if model_name=="lcbench":
    bound_fid=[1,50]
    fo =P_lcb( task_id=task_id,task=2,x_fid=x_fid)
else:
    bound_fid=[1,100]
    fo =P_NN( task_id=task_id,task=2,x_fid=x_fid)



dim=fo.d

max_N0=5 * dim

l = []
for parameter in ['r','v']:
    l.append(ContinuousParameter(parameter, -1, 1))

space = ParameterSpace(l)

s_min = 100
s_max = 50000


def wrapper(x, s):
    res=[]

    res.append({
        "function_value": sum(x**2),
        'cost':s,
    })


    return res[0]["function_value"], res[0]["cost"]


res = fmin_fabolas(wrapper, space=space, s_min=s_min, s_max=s_max, n_iters=100, marginalize_hypers=False)
