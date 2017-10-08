
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

def get_params_statistics(model, name1, name2):
    param_list = [named_param for named_param in model.named_parameters() if name1 in named_param[0] and name2 in named_param[0]]
    n_list = len(param_list)
    mean_list = np.zeros(n_list)
    std_list = np.zeros(n_list)
    for i_param, named_param in enumerate(param_list):
        param_name = named_param[0]
        param_vals = named_param[1]
        param_mean = param_vals.mean().data[0]
        param_std = param_vals.std().data[0]
        mean_list[i_param] = param_mean
        std_list[i_param] = param_std
        print('Parameter name: {}, mean value: {}, STD: {}'.format(param_name, param_mean, param_std))

    plot_statistics(mean_list, std_list, name1, name2)
    return mean_list, std_list

def plot_statistics(mean_list, std_list, name1, name2):
    plt.figure()
    n_list = len(mean_list)
    plt.errorbar(range(n_list), mean_list, yerr=std_list)
    plt.title("Statistics of the prior {} - {}".format(name1, name2))
    plt.xticks(np.arange(n_list))
    plt.xlabel('Layer')
    plt.ylabel('value')



log_var_w_mean, log_var_w_srd = get_params_statistics(prior_model, '_log_var', '.w_')
mu_w_mean, mu_w_srd = get_params_statistics(prior_model, '_mean', '.w_')

plt.show()

# log_var_params = [named_param for named_param in prior_model.named_parameters() if 'log_var' in named_param[0]]
#
# b_log_var = [named_param for named_param in log_var_params if '.b_' in named_param[0]]
# w_log_var = [named_param for named_param in log_var_params if '.w_' in named_param[0]]
#
# n_layers = len(b_log_var)
# b_log_var_mean = np.zeros(n_layers)
# b_log_var_std = np.zeros(n_layers)
#
#
#
# w_mean = np.zeros(n_layers)
# w_std = np.zeros(n_layers)
# for i_param, named_param in enumerate(w_log_var):
#     param_name = named_param[0]
#     param_vals = named_param[1]
#     param_mean =  param_vals.mean().data[0]
#     param_std = param_vals.std().data[0]
#     w_mean[i_param] = param_mean
#     w_std[i_param] = param_std
#     print('Parameter name: {}, mean value: {}, STD: {}'.format(param_name, param_mean, param_std))
#
#
#
# plt.figure()
# plt.errorbar(range(n_layers), b_log_var_mean, yerr=b_log_var_std)
# plt.title("Statistics of the prior log-var of the bias weights")
# plt.xticks(np.arange(n_layers))
# plt.xlabel('Layer')
# plt.ylabel('log-var param.')


