
from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.optim as optim

from Deterministic_Meta_Learning import meta_training_deterministic, meta_testing_deterministic
from Models.models_standard import get_model
from Single_Task import learn_single_standard
from Utils import common as cmn, data_gen
from Utils.common import save_models_dict, load_models_dict, set_random_seed

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
                    default=30)

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-2)

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
model_type = 'FcNet3' # 'FcNet' \ 'ConvNet'\ 'FcNet3'

# Loss criterion
loss_criterion = cmn.get_loss_criterion(prm.loss_type)

#  Define optimizer:
# optim_func, optim_args = optim.Adam,  {'lr': prm.lr,} #   'weight_decay': 1e-4
optim_func, optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10, 20]}
# lr_schedule = {} # No decay

# Meta-alg params:
prm.complexity_type = 'PAC_Bayes'   #  'Variational_Bayes' / 'PAC_Bayes' /
prm.hyper_prior_factor = 1e-5

init_from_prior = True  #  False \ True . In meta-testing -  init posterior from learned prior
# -------------------------------------------------------------------------------------------
# Generate the data sets of the training tasks:
# -------------------------------------------------------------------------------------------
n_train_tasks = 10
train_tasks_data = [data_gen.get_data_loader(prm) for i_task in range(n_train_tasks)]


# -------------------------------------------------------------------------------------------
#  Run Meta-Training
# -------------------------------------------------------------------------------------------

load_pretrained_prior = False  # False \ True
dir_path = './saved_prior'

if load_pretrained_prior:
    # Loads  previously training prior.
    # First, create the models:
    prior_dict ={'means_model': get_model(model_type, prm),
                 'log_var_model': get_model(model_type, prm)}
    # Then load the weights:
    load_models_dict(prior_dict, dir_path)
    print('Pre-trained  prior loaded from ' + dir_path)

else:
    # Meta-training to learn prior:
    prior_dict = meta_training_deterministic.run_meta_learning(train_tasks_data,
                                                               prm, model_type, optim_func, optim_args, loss_criterion, lr_schedule)
    # save learned prior:
    save_models_dict(prior_dict, dir_path)
    print('Trained prior saved in ' + dir_path)


# -------------------------------------------------------------------------------------------
# Generate the data sets of the test tasks:
# -------------------------------------------------------------------------------------------

n_test_tasks = 1
limit_train_samples = 1000
test_tasks_data = [data_gen.get_data_loader(prm, limit_train_samples) for _ in range(n_test_tasks)]

# -------------------------------------------------------------------------------------------
#  Run Meta-Testing
# -------------------------------------------------------------------------------

test_err_avg = 0
for i_task in range(n_test_tasks):
    print('Meta-Testing task {} out of {}...'.format(i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err = meta_testing_deterministic.run_learning(task_data, prior_dict, prm,
                                                       model_type, optim_func, optim_args, loss_criterion,
                                                       lr_schedule, init_from_prior)
    test_err_avg += test_err / n_test_tasks


# -------------------------------------------------------------------------------------------
#  Compare to standard learning
# -------------------------------------------------------------------------------------------

test_err_avg2 = 0
for i_task in range(n_test_tasks):
    print('Standard learning task {} out of {}...'.format(i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err, _ = learn_single_standard.run_learning(task_data, prm, model_type,
                                                  optim_func, optim_args, loss_criterion, lr_schedule)
    test_err_avg2 += test_err / n_test_tasks


# -------------------------------------------------------------------------------------------
#  Print results
# -------------------------------------------------------------------------------------------
cmn.write_result('-'*5 + ' Final Results: '+'-'*5, prm.log_file)
cmn.write_result('Meta-Testing - Avg test err: {0}%'.format(100 * test_err_avg), prm.log_file)
cmn.write_result('Standard - Avg test err: {0}%'.format(100 * test_err_avg2), prm.log_file)