
from __future__ import absolute_import, division, print_function
import torch

import torch.optim as optim

import timeit
import argparse

import data_gen
import models_standard
import common as cmn
# torch.backends.cudnn.benchmark=True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()


parser.add_argument('--data-source', type=str, help="Data: 'MNIST'",
                    default='MNIST')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=20)

parser.add_argument('--lr', type=float, help='learning rate',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--test-batch-size', help='input batch size for testing',
                    type=int, default=1000)

prm = parser.parse_args()
prm.cuda = not prm.no_cuda and torch.cuda.is_available()

torch.manual_seed(prm.seed)

#  Get model:
model_type = 'ConvNet' # 'FcNet' \ 'ConvNet'
model = models_standard.get_model(model_type, prm)

# Loss criterion
loss_criterion = cmn.get_loss_criterion(prm.loss_type)

#  Get optimizer:
optimizer = optim.Adam(model.parameters(), lr=prm.lr)

# Learning rate decay schedule:
lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10]}
# lr_schedule = {} # No decay

# -------------------------------------------------------------------------------------------
#  Run learning
# -------------------------------------------------------------------------------------------
from learn_standard import run_learning
run_learning(prm, model, optimizer, loss_criterion, lr_schedule)