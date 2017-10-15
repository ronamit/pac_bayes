
from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.optim as optim

from Utils import common as cmn, data_gen
from Utils.common import set_random_seed
from Single_Task import learn_single_Bayes

# torch.backends.cudnn.benchmark=True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / Omniglot",
                    default='MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels' ",
                    default='None')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=50) # 300

parser.add_argument('--lr', type=float, help='learning rate (initial)',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=666)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=1000)

parser.add_argument('--log-file', type=str, help='Name of file to save log (None = no save)',
                    default='log')

prm = parser.parse_args()
prm.cuda = True

prm.data_path = '../data'

set_random_seed(prm.seed)

# For Omniglot data - N = number of classes. K = number of train samples per class:
# Note: number of test samples per class is 20-K
if prm.data_source == 'Omniglot':
    prm.n_way_k_shot = {'N': 5, 'K': 10}

#  Get model:
prm.model_name = 'ConvNet'   # 'FcNet2' / 'FcNet3' / 'ConvNet' / 'ConvNet_Dropout'

# Weights initialization:
prm.bayes_inits = {'Bayes-Mu': {'bias': 0, 'std': 0.1}, 'Bayes-log-var': {'bias': -10, 'std': 0.1}}

# None = use default initializer
# Note:
# 1. start with small sigma - so gradients variance estimate will be low

# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 3

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr}
# optim_func, optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}


# Learning rate decay schedule:
# lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10, 30]}
prm.lr_schedule = {} # No decay

# Learning parameters:
# In the stage 1 of the learning epochs, epsilon std == 0
# In the second stage it increases linearly until reaching std==1 (full eps)
prm.stage_1_ratio = 0  # 0.05
prm.full_eps_ratio_in_stage_2 = 0.5 # 0.5


# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote'

# Generate task data set:
limit_train_samples = None  # None
data_loader = data_gen.get_data_loader(prm, limit_train_samples=limit_train_samples)

# -------------------------------------------------------------------------------------------
#  Run learning
# -------------------------------------------------------------------------------------------

learn_single_Bayes.run_learning(data_loader, prm)