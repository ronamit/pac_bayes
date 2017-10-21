
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from Stochsastic_Meta_Learning import meta_test_Bayes, meta_train_Bayes
from Models.models import get_model
from Single_Task import learn_single_Bayes, learn_single_standard
from Utils import data_gen
from Utils.common import save_model_state, load_model_state, write_result, set_random_seed
from Utils.Bayes_utils import kld_element

# torch.backends.cudnn.benchmark=True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7


# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'Sinusoid' ",
                    default='MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation: 'None' / 'Permute_Pixels' / 'Permute_Labels'",
                    default='Permute_Pixels')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=300) # 200

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=1000)

parser.add_argument('--log-file', type=str, help='Name of file to save log (default: no save)',
                    default='log')

prm = parser.parse_args()
prm.cuda = True

prm.data_path = '../data'

set_random_seed(prm.seed)

#  Define model type (hypothesis class):
prm.model_name = 'ConvNet'   # 'FcNet2' / 'FcNet3' / 'ConvNet' / 'ConvNet_Dropout'

# Weights initialization (for Bayesian net):
prm.bayes_inits = {'Bayes-Mu': {'bias': 0, 'std': 0.1}, 'Bayes-log-var': {'bias': -10, 'std': 0.1}}
# Note:
# 1. start with small sigma - so gradients variance estimate will be low
# 2.  don't init with too much std so that complexity term won't be too large

# Weights initialization (for standard comparision net):
prm.init_override = None # None = use default initializer
# prm.init_override = {'mean': 0, 'std': 0.1}


# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 3

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr} #'weight_decay': 1e-4
# optim_func, optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
#lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [150]}
prm.lr_schedule = {} # No decay

# Meta-alg params:
prm.complexity_type = 'PAC_Bayes_Seeger'
#  'Variational_Bayes' / 'PAC_Bayes_McAllaster' / 'PAC_Bayes_Pentina' / 'PAC_Bayes_Seeger'  / 'KLD' / 'NoComplexity'

init_from_prior = True  #  False \ True . In meta-testing -  init posterior from learned prior

# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote' / 'AvgVote'

# -------------------------------------------------------------------------------------------
#  Load pr-trained prior
# -------------------------------------------------------------------------------------------

dir_path = './saved'
f_name = 'pror_Permuted_labels_MNIST_ConvNet'  #  /


# Loads  previously training prior.
# First, create the model:
prior_model = get_model(prm, 'Stochastic')
# Then load the weights:
load_model_state(prior_model, dir_path, name=f_name)
print('Pre-trained  prior loaded from ' + dir_path)
# -------------------------------------------------------------------------------------------
# Generate the data sets of the test tasks:
# -------------------------------------------------------------------------------------------

limit_train_samples = 2000

print('-'*5 + 'Generating 1 test-task with at most {} training samples'.format(limit_train_samples)+'-'*5)

test_task_data = data_gen.get_data_loader(prm, limit_train_samples)

# --------------------------------------------------------------------------------
#  Run Meta-Testing
# -------------------------------------------------------------------------------
print('Meta-Testing with transferred prior....')

test_err, post_model = meta_test_Bayes.run_learning(
    test_task_data, prior_model, prm, init_from_prior, verbose=0)

# --------------------------------------------------------------------------------
#  Analyze
# -------------------------------------------------------------------------------


prior_layers_list = list(prior_model.children())
post_layers_list = list(post_model.children())
n_layer = len(prior_layers_list)
w_kld_per_layer = np.zeros(n_layer)
b_kld_per_layer = np.zeros(n_layer)
all_weight_kld_per_layer = np.zeros(n_layer)


total_kld = 0
for i_layer, prior_layer in enumerate(prior_layers_list):
    post_layer = post_layers_list[i_layer]

    w_kld_per_layer[i_layer] = kld_element(post_layer.w, prior_layer.w).data[0]
    b_kld_per_layer[i_layer] = kld_element(post_layer.b, prior_layer.b).data[0]

    all_weight_kld_per_layer[i_layer] = w_kld_per_layer[i_layer] + b_kld_per_layer[i_layer]

    # normalize by the number of weights:
    w_numel = prior_layer.in_dim * prior_layer.out_dim
    b_numel = prior_layer.out_dim
    w_kld_per_layer[i_layer] /= w_numel
    b_kld_per_layer[i_layer] /= b_numel
    all_weight_kld_per_layer[i_layer] /= w_numel + b_numel



def plot_layes_kld(kld_per_layer, title_str):
    plt.figure()
    plt.plot(range(n_layer), kld_per_layer)
    plt.title(title_str)
    plt.xticks(np.arange(n_layer))
    plt.xlabel('Layer')
    plt.ylabel('value')


plot_layes_kld(all_weight_kld_per_layer, "Mean KLD between posterior and prior per layer")


plt.show()