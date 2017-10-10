
from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.optim as optim

from Deterministic_Meta_Learning import meta_train_det, meta_test_det
from Models.models_standard import get_model
from Single_Task import learn_single_standard
from Utils import common as cmn, data_gen
from Utils.common import save_models_dict, load_models_dict, set_random_seed, write_result

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
                    default=40)

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
model_type = 'FcNet' # 'FcNet' \ 'ConvNet'\ 'FcNet3'

# Weights initialization:
prm.inits ={'Bayes-Mu': {'bias': 0, 'std': 0.1},
            'Bayes-log-var': {'bias': -10, 'std': 0.1},
            'Standard-Net': {'bias': 0, 'std': 0.1}}
# None = use default initializer
# Note:
# 1.  don't init with too much std so that complexity term won't be too large

# Loss criterion
prm.loss_criterion = cmn.get_loss_criterion(prm.loss_type)

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr,} #   'weight_decay': 1e-4
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10, 20]}
prm.lr_schedule = {} # No decay

# Meta-alg params:
prm.complexity_type = 'Variational_Bayes'   #  'Variational_Bayes' / 'PAC_Bayes_McAllaster' / 'KLD' / 'NoComplexity' / 'PAC_Bayes_Pentina'
print(prm.complexity_type)
prm.hyper_prior_factor = 1e-6

init_from_prior = True  #  False \ True . In meta-testing -  init posterior from learned prior
# -------------------------------------------------------------------------------------------
# Generate the data sets of the training tasks:
# -------------------------------------------------------------------------------------------
n_train_tasks = 10

write_result('-'*5 + 'Generating {} training-tasks'.format(n_train_tasks)+'-'*5, prm.log_file)

train_tasks_data = [data_gen.get_data_loader(prm) for i_task in range(n_train_tasks)]

# -------------------------------------------------------------------------------------------
#  Run Meta-Training
# -------------------------------------------------------------------------------------------

mode = 'MetaTrain'  # 'MetaTrain'  \ 'LoadPrior' \ 'FromScratch'
dir_path = './saved'

if mode == 'MetaTrain':
    # Meta-training to learn prior:
    prior_dict = meta_train_det.run_meta_learning(train_tasks_data, prm, model_type)

    # save learned prior:
    save_models_dict(prior_dict, dir_path)
    print('Trained prior saved in ' + dir_path)

elif mode == 'LoadPrior':
    # Loads  previously training prior.
    # First, create the models:
    prior_dict ={'means_model': get_model(model_type, prm),
                 'log_var_model': get_model(model_type, prm)}
    # Then load the weights:
    load_models_dict(prior_dict, dir_path)
    print('Pre-trained  prior loaded from ' + dir_path)

else:
    prior_dict = None

# -------------------------------------------------------------------------------------------
# Generate the data sets of the test tasks:
# -------------------------------------------------------------------------------------------

n_test_tasks = 5
limit_train_samples = 2000

write_result('-'*5 + 'Generating {} test-tasks with at most {} training samples'.
             format(n_test_tasks, limit_train_samples)+'-'*5, prm.log_file)

test_tasks_data = [data_gen.get_data_loader(prm, limit_train_samples) for _ in range(n_test_tasks)]

# -------------------------------------------------------------------------------------------
#  Run Meta-Testing
# -------------------------------------------------------------------------------


test_err_avg_mt = 0
for i_task in range(n_test_tasks):
    print('Meta-Testing task {} out of {}...'.format(i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err = meta_test_det.run_learning(task_data, prior_dict, prm,
                                          model_type, init_from_prior=init_from_prior, verbose=0)
    test_err_avg_mt += test_err / n_test_tasks


# -------------------------------------------------------------------------------------------
#  Compare to standard learning
# -------------------------------------------------------------------------------------------

test_err_sd = 0
for i_task in range(n_test_tasks):
    print('Standard learning task {} out of {}...'.format(i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err, _ = learn_single_standard.run_learning(task_data, prm, model_type, verbose=0)
    test_err_sd += test_err / n_test_tasks


# -------------------------------------------------------------------------------------------
#  Print results
# -------------------------------------------------------------------------------------------
write_result('-'*5 + ' Final Results: '+'-'*5, prm.log_file)
write_result('Meta-Testing - Avg test err: {0}%'.format(100 * test_err_avg_mt), prm.log_file)
write_result('Standard - Avg test err: {0}%'.format(100 * test_err_sd), prm.log_file)