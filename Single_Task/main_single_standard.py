
from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.optim as optim

from Utils import common as cmn, data_gen
from Utils.common import set_random_seed
from Single_Task import learn_single_standard

# torch.backends.cudnn.benchmark=True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST'",
                    default='MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'",
                    default='None')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=20)

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=1000)

parser.add_argument('--log-file', type=str, help='Name of file to save log (default: no save)',
                    default='log')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')


prm = parser.parse_args()
prm.cuda = not prm.no_cuda and torch.cuda.is_available()

prm.data_path = '../data'

set_random_seed(prm.seed)

#  Define model:
model_type = 'FcNet' # 'FcNet' \ 'ConvNet'

# Weights initialization (for standard models):
# None = use default initializer
prm.weights_init_std = None
prm.weights_init_bias = None

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr}
# optim_func, optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
#lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10]}
prm.lr_schedule = {} # No decay

# Generate task data set:
data_loader = data_gen.get_data_loader(prm)

# -------------------------------------------------------------------------------------------
#  Run learning
# -------------------------------------------------------------------------------------------

learn_single_standard.run_learning(data_loader, prm, model_type)