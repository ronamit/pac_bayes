
from __future__ import absolute_import, division, print_function

import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from Models.models import get_model
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

set_random_seed(prm.seed)


# Weights initialization (for Bayesian net):
prm.bayes_inits = {'Bayes-Mu': {'bias': 0, 'std': 0.1}, 'Bayes-log-var': {'bias': -10, 'std': 0.1}}


# -------------------------------------------------------------------------------------------
#  Load pr-trained prior
# -------------------------------------------------------------------------------------------

dir_path = './saved'

name =  'Permuted_labels'  # 'PermutedPixels' / Permuted_labels

if name == 'PermutedPixels':
    # Permute Pixels:
    file_name_prior = 'prior_PermutedPixels_MNIST_FCNet3'  #
    prm.model_name = 'FcNet3'
    layers_names = ('FC1', 'FC2', 'FC3', 'FC_out')
    # ***************
else:
    # Permute Labels:
    file_name_prior = 'pror_Permuted_labels_MNIST'
    prm.model_name = 'ConvNet'
    layers_names = ('conv1', 'conv2', 'FC1', 'FC_out')



# Loads  previously training prior.
# First, create the model:
prior_model = get_model(prm, 'Stochastic')
# Then load the weights:
is_loaded = load_model_state(prior_model, dir_path, name=file_name_prior)
if not is_loaded:
    raise ValueError('No prior found in the name: ' + file_name_prior)
print('Pre-trained  prior loaded from ' + dir_path)
# -------------------------------------------------------------------------------------------


def extract_param_list(model, name1, name2):
    return [named_param for named_param in model.named_parameters() if name1 in named_param[0] and name2 in named_param[0]]


w_mu_params = extract_param_list(prior_model,'_mean', '.w_')
b_mu_params = extract_param_list(prior_model,'_mean', '.b_')
w_log_var_params = extract_param_list(prior_model,'_log_var', '.w_')
b_log_var_params = extract_param_list(prior_model,'_log_var', '.b_')

n_layers = len(w_mu_params)

def log_var_to_sigma(log_var_params):
    return [(named_param[0].replace('_log_var', '_sigma'),
              0.5 * torch.exp(named_param[1]))
             for named_param in log_var_params]

w_sigma_params = log_var_to_sigma(w_log_var_params)
b_sigma_params = log_var_to_sigma(b_log_var_params)


def get_params_statistics(param_list):
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
        print('Parameter name: {}, mean value: {:.3}, STD: {:.3}'.format(param_name, param_mean, param_std))
    return mean_list, std_list


def plot_statistics(mean_list, std_list, name):
    plt.figure()
    n_list = len(mean_list)
    plt.errorbar(range(n_list), mean_list, yerr=std_list)
    plt.title("Statistics of the prior {} ".format(name))
    plt.xticks(np.arange(n_list))
    plt.xlabel('Layer')
    plt.ylabel('value')

plot_statistics(*get_params_statistics(w_log_var_params), name='weights log-var')
plt.xticks( np.arange(len(layers_names)), layers_names)

plt.show()


# get_params_statistics(w_sigma_params)
# get_params_statistics(w_mu_params)

# def calc_SNR(mu_params, sigma_params):
#     w_snr = []
#     for i_layer in range(n_layers):
#         named_param = (mu_params[i_layer][0].replace('_mean', '_SNR'),
#                        torch.abs(mu_params[i_layer][1] / sigma_params[i_layer][1]))
#         w_snr.append(named_param)
#     return w_snr

# get_params_statistics(calc_SNR(w_mu_params, w_sigma_params))
# get_params_statistics(calc_SNR(b_mu_params, b_sigma_params))