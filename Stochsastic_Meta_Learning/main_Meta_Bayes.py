
from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim

from Stochsastic_Meta_Learning import meta_test_Bayes, meta_train_Bayes

from Models.stochastic_models import get_model
from Single_Task import learn_single_standard
from Utils.data_gen import get_data_loader
from Utils.common import save_model_state, load_model_state, write_result, set_random_seed

torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs
# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST'",
                    default='MNIST')

parser.add_argument('--n_train_tasks', type=int, help='Number of meta-training tasks',
                    default=5)

parser.add_argument('--data-transform', type=str, help="Data transformation",
                    default='Permute_Labels') #  'None' / 'Permute_Pixels' / 'Permute_Labels'

parser.add_argument('--loss-type', type=str, help="Loss function",
                    default='CrossEntropy') #  'CrossEntropy' / 'L2_SVM'

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='ConvNet3')  # ConvNet3 / 'FcNet3'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--n_meta_train_epochs', type=int, help='number of epochs to train',
                    default=300)  # 10 / 300
parser.add_argument('--n_meta_test_epochs', type=int, help='number of epochs to train',
                    default=300)  # 10 / 300

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=1000)

parser.add_argument('--log-file', type=str, help='Name of file to save log (None = no save)',
                    default='log')

prm = parser.parse_args()

prm.data_path = '../data'

set_random_seed(prm.seed)


# Weights initialization (for Bayesian net):
prm.bayes_inits = {'Bayes-Mu': {'bias': 0, 'std': 0.1}, 'Bayes-log-var': {'bias': -10, 'std': 0.1}}
# Note:
# 1. start with small sigma - so gradients variance estimate will be low
# 2.  don't init with too much std so that complexity term won't be too large

# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 1

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr} #'weight_decay': 1e-4
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [50, 150]}
prm.lr_schedule = {}  # No decay

# Meta-alg params:
prm.complexity_type = 'NewBound'
#  'Variational_Bayes' / 'PAC_Bayes_McAllaster' / 'PAC_Bayes_Pentina' / 'PAC_Bayes_Seeger'  / 'KLD' / 'NoComplexity'

prm.hyperprior_factor = 1e-7  # multiplicative  factor for the hyper-prior regularization
prm.kappa_factor = 0  #
prm.delta = 0.1  #  maximal probability that the bound does not hold

init_from_prior = True  #  False \ True . In meta-testing -  init posterior from learned prior

prm.meta_batch_size = 5  # how many tasks in each meta-batch

# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote' / 'AvgVote'

# -------------------------------------------------------------------------------------------
#  Run Meta-Training
# -------------------------------------------------------------------------------------------

mode = 'MetaTrain'  # 'MetaTrain'  \ 'LoadPrior' \
dir_path = './saved'
file_name = 'prior'

if mode == 'MetaTrain':

    n_train_tasks = prm.n_train_tasks
    # Generate the data sets of the training tasks:
    write_result('-' * 5 + 'Generating {} training-tasks'.format(n_train_tasks) + '-' * 5, prm.log_file)
    train_data_loaders = [get_data_loader(prm, meta_split='meta_train') for i_task in range(n_train_tasks)]

    # Meta-training to learn prior:
    prior_model = meta_train_Bayes.run_meta_learning(train_data_loaders, prm)
    # save learned prior:
    f_path = save_model_state(prior_model, dir_path, name=file_name)
    print('Trained prior saved in ' + f_path)


elif mode == 'LoadPrior':

    # Loads  previously training prior.
    # First, create the model:
    prior_model = get_model(prm)
    # Then load the weights:
    load_model_state(prior_model, dir_path, name=file_name)
    print('Pre-trained  prior loaded from ' + dir_path)
else:
    raise ValueError('Invalid mode')

# -------------------------------------------------------------------------------------------
# Generate the data sets of the test tasks:
# -------------------------------------------------------------------------------------------

n_test_tasks = 10
limit_train_samples = 2000

write_result('-'*5 + 'Generating {} test-tasks with at most {} training samples'.
             format(n_test_tasks, limit_train_samples)+'-'*5, prm.log_file)

test_tasks_data = [get_data_loader(prm, limit_train_samples=limit_train_samples, meta_split='meta_test') for _ in range(n_test_tasks)]

# -------------------------------------------------------------------------------------------
#  Run Meta-Testing
# -------------------------------------------------------------------------------
write_result('Meta-Testing with transferred prior....', prm.log_file)

test_err_bayes = np.zeros(n_test_tasks)
for i_task in range(n_test_tasks):
    print('Meta-Testing task {} out of {}...'.format(1+i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err_bayes[i_task], _ = meta_test_Bayes.run_learning(task_data, prior_model, prm, init_from_prior, verbose=0)


# -------------------------------------------------------------------------------------------
#  Compare to standard learning
# -------------------------------------------------------------------------------------------

write_result('Run standard learning from scratch....', prm.log_file)
test_err_standard = np.zeros(n_test_tasks)
prm_standard = deepcopy(prm)
prm_standard.num_epochs = prm.n_meta_test_epochs
for i_task in range(n_test_tasks):
    print('Standard learning task {} out of {}...'.format(i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err_standard[i_task], _ = learn_single_standard.run_learning(task_data, prm_standard, verbose=0)


# -------------------------------------------------------------------------------------------
#  Print results
# -------------------------------------------------------------------------------------------
write_result('-'*5 + ' Final Results: '+'-'*5, prm.log_file)
write_result('Meta-Testing - Avg test err: {:.3}%, STD: {:.3}%'
             .format(100 * test_err_bayes.mean(), 100 * test_err_bayes.std()), prm.log_file)
write_result('Standard - Avg test err: {:.3}%, STD: {:.3}%'.
             format(100 * test_err_standard.mean(), 100 * test_err_standard.std()), prm.log_file)

# -------------------------------------------------------------------------------------------
#  Print prior analysis
# -------------------------------------------------------------------------------------------
from Stochsastic_Meta_Learning.Analyze_Prior import run_prior_analysis
run_prior_analysis(prior_model)
