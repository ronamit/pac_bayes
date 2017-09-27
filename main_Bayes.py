
from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.optim as optim

import common as cmn
import models_Bayes

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
                    default=500)

parser.add_argument('--lr', type=float, help='learning rate (initial)',
                    default=1e-2)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size', type=int, help='input batch size for testing',
                    default=1000)

prm = parser.parse_args()
prm.cuda = True

torch.manual_seed(prm.seed)

# Weights initialization:
prm.rand_init_std = 0.01

#  Get model:
model_type = 'BayesNN' # 'BayesNN' \ 'BigBayesNN'
prm.model_type = model_type
model = models_Bayes.get_model(model_type, prm)

#  Get optimizer:
optimizer = optim.Adam(model.parameters(), lr=prm.lr)
# optimizer = optim.SGD(model.parameters(), lr=prm.lr, momentum=0.9)

# Learning rate decay schedule:
lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10, 100]}
# lr_schedule = {} # No decay

# Loss criterion:
loss_criterion = cmn.get_loss_criterion(prm.loss_type)

# -------------------------------------------------------------------------------------------
#  Run learning
# -------------------------------------------------------------------------------------------

from learn_Bayes import run_learning
run_learning(prm, model, optimizer, loss_criterion, lr_schedule)