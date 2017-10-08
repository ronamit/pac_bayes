from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.optim as optim

from Stochsastic_Meta_Learning import meta_testing_Bayes, meta_training_Bayes
from Models import models_Bayes
from Single_Task import learn_single_Bayes, learn_single_standard
from Utils import data_gen
from Utils.common import save_model_state, load_model_state, get_loss_criterion, write_result, set_random_seed

import Single_Task.learn_single_standard
# torch.backends.cudnn.benchmark=True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7



# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'Sinusoid' ",
                    default='MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation: 'None' / 'Permute_Pixels' / 'Permute_Labels'",
                    default='Permute_Labels')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=200) # 200

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

#  Define model:
model_type = 'FcNet'

# Loss criterion
prm.loss_criterion = get_loss_criterion(prm.loss_type)


#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr}
# optim_func, optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10]}
prm.lr_schedule = {} # No decay

# Generate the task 1 data set:
task1_data = data_gen.get_data_loader(prm)



#  Run learning of task 1
test_err, model = learn_single_standard.run_learning(
    task1_data, prm)

# Generate the task 2 data set:
task2_data = data_gen.get_data_loader(prm)


#  Run learning of task 2 from scratch:
learn_single_standard.run_learning(task1_data, model_type, prm)

