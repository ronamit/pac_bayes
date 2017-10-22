
from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.optim as optim

from Utils import common as cmn, data_gen
from Utils.common import set_random_seed, save_model_state
from Single_Task import learn_single_standard

torch.backends.cudnn.benchmark=True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'CIFAR10' / Omniglot",
                    default='CIFAR10')

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'",
                    default='None')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=50)

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-1)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=128)

parser.add_argument('--log-file', type=str, help='Name of file to save log (None = no save)',
                    default='log')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')


prm = parser.parse_args()
prm.cuda = True

prm.data_path = '../data'

set_random_seed(prm.seed)

# For Omniglot data - N = number of classes. K = number of train samples per class:
# Note: number of test samples per class is 20-K
# prm.n_way_k_shot = {'N': 10, 'K': 5}

#  Define model:
prm.model_name = 'DenseNet20'   # 'FcNet2' / 'FcNet3' / 'ConvNet' / 'ConvNet_Dropout' / 'OmniglotNet' / WideResNet / DenseNet  / DenseNet60 / DenseNet100/ DenseNet20

# Weights initialization:
prm.init_override = None # None = use default initializer
# prm.init_override = {'mean': 0, 'std': 0.1}

#  Define optimizer:
# prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr, 'weight_decay':5e-4}
prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9, 'weight_decay':5e-4}

# Learning rate decay schedule:
prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [150, 225]}
# prm.lr_schedule = {} # No decay

# Generate task data set:
limit_train_samples = None  # None
data_loader = data_gen.get_data_loader(prm, limit_train_samples=limit_train_samples)

# -------------------------------------------------------------------------------------------
#  Run learning
# -------------------------------------------------------------------------------------------

test_err, model = learn_single_standard.run_learning(data_loader, prm)

dir_path = './saved'
f_name='standard_weights'
f_path = save_model_state(model, dir_path, name=f_name)
print('Trained model saved in ' + f_path)