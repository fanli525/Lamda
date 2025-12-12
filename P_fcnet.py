import numpy as np
from yahpo_gym import *
import copy
import logging

logging.basicConfig(level=logging.ERROR)
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter, ConfigurationSpace


def get_vartypes(configspace):
    hps =configspace.get_hyperparameters()

    kde_vartypes = ""
    vartypes = []


    for h in hps:
        if hasattr(h, 'sequence'):
            raise RuntimeError(
                'This version on BOHB does not support ordinal hyperparameters. Please encode %s as an integer parameter!' % (
                    h.name))

        if hasattr(h, 'choices'):
            kde_vartypes += 'u'
            vartypes += [len(h.choices)]
        else:
            kde_vartypes += 'c'
            vartypes += [0]

    vartypes = np.array(vartypes, dtype=int)
    return kde_vartypes, vartypes



def get_hyperparameter_search_space():
    cs = ConfigurationSpace()
    batch_size = UniformIntegerHyperparameter(
        "batch_size", 8, 64, log=True, default_value=64)
    init_lr = UniformFloatHyperparameter(
        "init_lr", 5e-4, 1e-1, log=True, default_value=1e-3)
    dropout_1 = UniformFloatHyperparameter(
        "dropout_1", 0.0, 0.6, log=False, default_value=0.5)
    dropout_2 = UniformFloatHyperparameter(
        "dropout_2", 0.0, 0.6, log=False, default_value=0.5)
    n_units_1 = UniformIntegerHyperparameter(
        "n_units_1", 16, 512, log=False, default_value=256)
    n_units_2 = UniformIntegerHyperparameter(
        "n_units_2", 16, 512, log=False, default_value=256)
    cs.add_hyperparameters(
        [batch_size, init_lr, dropout_1, dropout_2, n_units_1, n_units_2])
    return cs


class P_NN:

    def __init__(self, task_id, alg_name='bohb', x_fid=[]):
        self.alg_name = alg_name
        model_name = "fcnet"
        b = BenchmarkSet(scenario=model_name)
        b.set_instance(task_id)
        opt_space = b.get_opt_space()
        self.task_id = task_id
        a = b.get_opt_space()
        a1 = a.get_hyperparameter_names()

        self.x_fid = x_fid
        self.task = len(x_fid)
        fidelity_space = b.get_fidelity_space()
        d_int = []
        d_float = []
        d_log = []
        bound_int = []
        bound_float = []
        for i in range(len(a1)):
            try:
                if isinstance(a.get_hyperparameter(a1[i]).lower, int):
                    d_int.append(i)
                    if a.get_hyperparameter(a1[i]).name != 'epoch':
                        bound_int.append([a.get_hyperparameter(a1[i]).lower, a.get_hyperparameter(a1[i]).upper])
                if isinstance(a.get_hyperparameter(a1[i]).lower, float):
                    d_float.append(i)
                    bound_float.append([a.get_hyperparameter(a1[i]).lower, a.get_hyperparameter(a1[i]).upper])
                if a.get_hyperparameter(a1[i]).log == True:
                    d_log.append(i)
            except:
                pass
        self.cs_name = ['batch_size', 'dropout_1', 'dropout_2', 'init_lr',
                        'n_units_1', 'n_units_2']
        bound = []
        for i in self.cs_name:
            bound.append([a.get_hyperparameter(i).lower, a.get_hyperparameter(i).upper])

        bound = np.array(bound)
        self.x_name = a1

        bound_fid = [1, 100]
        self.bound_fid = bound_fid
        self.bound = bound

        self.d_float = [1, 2, 3]
        self.d_int = [0, 4, 5]
        self.d_only_int = [4, 5]
        self.d_log = [0, 4]
        self.b = b
        self.bound_int = bound[self.d_int, :]
        self.bound_float = bound[self.d_float, :]
        self.d = len(self.d_int) + len(self.d_float)

        self.X = []
        self.f = []
        self.fidelity = []
        self.cost_all = []

    def fit(self, config, budget):
        if self.alg_name in ['mumbo', 'sgp_prior', 'mumbo_prior', 'sgp', 'pbf', 'turbo', 'dnn_mfbo', 'dnn_mfbo_prior']:
            hps = self.b.config_space.sample_configuration()
            for i in range(len(self.cs_name)):
                try:
                    if i in self.d_int:
                        hps[self.cs_name[i]] = int(config[i])
                    else:
                        hps[self.cs_name[i]] = config[i]
                except:
                    pass
        else:
            hps = self.b.config_space.sample_configuration()
            hps['activation_fn_1'] = 'relu'
            hps['activation_fn_2'] = 'relu'
            hps['lr_schedule'] = 'const'
            hps['task'] = self.task_id
            hps['replication'] = 1

            for key in config.keys():
                hps[key] = config[key]
        hps['epoch'] = int(budget)
        f = self.b.objective_function(hps)[0]['valid_loss']
        cost = self.b.objective_function(hps)[0]['runtime']

        if self.alg_name in ['mumbo', 'sgp_prior', 'mumbo_prior', 'sgp', 'pbf', 'turbo', 'dnn_mfbo', 'dnn_mfbo_prior']:
            xnew = list(config)
        else:
            xnew = list(config.values())

        self.X.append(xnew)
        self.f.append(f)
        self.fidelity.append(budget)
        self.cost_all.append(cost)

        if self.alg_name in ['bohb', 'bohb2', 'bohb_prior', 'bohb_prior_adp', 'bohb_hf', 'bohb_all', 'hyperband']:
            return ({
                'loss': float(f),
                'info': float(cost)})
        elif self.alg_name in ['turbo']:
            return f
        else:
            return f, cost

    def __call__(self, config):
        budget = 100
        if self.alg_name in ['mumbo', 'sgp_prior', 'mumbo_prior', 'sgp', 'turbo', 'dnn_mfbo', 'dnn_mfbo_prior']:
            hps = self.b.config_space.sample_configuration()
            for i in range(len(self.cs_name)):
                try:
                    if i in self.d_int:
                        hps[self.cs_name[i]] = int(config[i])
                    else:
                        hps[self.cs_name[i]] = config[i]
                except:
                    pass
        else:
            hps = self.b.config_space.sample_configuration()
            hps['activation_fn_1'] = 'relu'
            hps['activation_fn_2'] = 'relu'
            hps['lr_schedule'] = 'const'
            hps['task'] = self.task_id
            hps['replication'] = 1

            for key in config.keys():
                hps[key] = config[key]
        hps['epoch'] = int(budget)
        f = self.b.objective_function(hps)[0]['valid_loss']
        cost = self.b.objective_function(hps)[0]['runtime']
        if self.alg_name in ['mumbo', 'sgp_prior', 'mumbo_prior', 'sgp', 'turbo', 'dnn_mfbo', 'dnn_mfbo_prior']:
            xnew = list(config)
        else:
            xnew = list(config.values())
        self.X.append(xnew)
        self.f.append(f)
        self.fidelity.append(budget)
        self.cost_all.append(cost)

        return f
