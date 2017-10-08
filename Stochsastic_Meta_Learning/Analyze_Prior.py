
from __future__ import absolute_import, division, print_function

import argparse


import numpy as np
import matplotlib.pyplot as plt

from Models import models_Bayes
from Utils.common import save_model_state, load_model_state, write_result, set_random_seed

#  settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'Sinusoid' ",
                    default='MNIST')

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--log-file', type=str, help='Name of file to save log (default: no save)',
                    default='log')

prm = parser.parse_args()
prm.cuda = True

set_random_seed(prm.seed)


dir_path = './saved'
f_name='prior_PermuteLabels'

model_type = 'BayesNN'  # 'BayesNN' \ 'BigBayesNN'
#  TODO: get prm from file

# Weights initialization:
prm.log_var_init_std = 0.1
prm.log_var_init_bias = -10
prm.mu_init_std = 0.1
prm.mu_init_bias = 0.0

# Loads  previously training prior.
# First, create the model:
prior_model = models_Bayes.get_bayes_model(model_type, prm)
# Then load the weights:
load_model_state(prior_model, dir_path, name=f_name)


log_var_params = [named_param for named_param in prior_model.named_parameters() if 'log_var' in named_param[0]]

b_log_var = [named_param for named_param in log_var_params if '.b_' in named_param[0]]
w_log_var = [named_param for named_param in log_var_params if '.w_' in named_param[0]]

n_layers = len(b_log_var)
b_mean = np.zeros(n_layers)
b_std = np.zeros(n_layers)
for i_param, named_param in enumerate(b_log_var):
    param_name = named_param[0]
    param_vals = named_param[1]
    param_mean =  param_vals.mean().data[0]
    param_std = param_vals.std().data[0]
    b_mean[i_param] = param_mean
    b_std[i_param] = param_std
    print('Parameter name: {}, mean value: {}, STD: {}'.format(param_name, param_mean, param_std))

w_mean = np.zeros(n_layers)
w_std = np.zeros(n_layers)
for i_param, named_param in enumerate(w_log_var):
    param_name = named_param[0]
    param_vals = named_param[1]
    param_mean =  param_vals.mean().data[0]
    param_std = param_vals.std().data[0]
    w_mean[i_param] = param_mean
    w_std[i_param] = param_std
    print('Parameter name: {}, mean value: {}, STD: {}'.format(param_name, param_mean, param_std))


plt.figure()
plt.errorbar(range(n_layers), w_mean, yerr=w_std)
plt.title("Statistics of the prior log-var of the multiplication weights")
plt.xticks(np.arange(n_layers))
plt.xlabel('Layer')
plt.ylabel('log-var param.')


plt.figure()
plt.errorbar(range(n_layers), b_mean, yerr=b_std)
plt.title("Statistics of the prior log-var of the bias weights")
plt.xticks(np.arange(n_layers))
plt.xlabel('Layer')
plt.ylabel('log-var param.')


plt.show()