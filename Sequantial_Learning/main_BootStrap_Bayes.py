
from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from Stochsastic_Meta_Learning import meta_test_Bayes, meta_train_Bayes_finite_tasks
from Models.models import get_model
from Single_Task import learn_single_Bayes, learn_single_standard
from Utils.data_gen import get_data_loader
from Utils.common import save_model_state, load_model_state, write_result, set_random_seed

torch.backends.cudnn.benchmark=True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST'",
                    default='MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation: 'None' / 'Permute_Pixels' / 'Permute_Labels'",
                    default='Permute_Pixels')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=50)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=300)

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=1000)

parser.add_argument('--log-file', type=str, help='Name of file to save log (None = no save)',
                    default='log')

prm = parser.parse_args()

from Data_Path import get_data_path
prm.data_path = get_data_path()

set_random_seed(prm.seed)


#  Define model type (hypothesis class):
prm.model_name = 'FcNet3'   # 'FcNet2' / 'FcNet3' / 'ConvNet' / 'ConvNet_Dropout'

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
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [50, 150]}
prm.lr_schedule = {} # No decay

# Meta-alg params:
prm.complexity_type = 'PAC_Bayes_McAllaster'
#  'Variational_Bayes' / 'PAC_Bayes_McAllaster' / 'PAC_Bayes_Pentina' / 'PAC_Bayes_Seeger'  / 'KLD' / 'NoComplexity'

prm.hyper_prior_factor = 1e-7  #  1e-5
# Note: Hyper-prior is important to keep the sigma not too low.
# Choose the factor  so that the Hyper-prior  will be in the same order of the other terms.

init_from_prior = True  #  False \ True . In meta-testing -  init posterior from learned prior

prm.meta_batch_size = 30  # how many tasks in each meta-batch

# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote' / 'AvgVote'

# -------------------------------------------------------------------------------------------
#  Run Meta-Training
# -------------------------------------------------------------------------------------------

mode = 'MetaTrain'  # 'MetaTrain'  \ 'LoadPrior' \
dir_path = './saved'
f_name = 'prior'


if mode == 'MetaTrain':

    # Generate the data sets of the training tasks:
    n_train_tasks = 5
    limit_train_samples = 10000
    write_result('-' * 5 + 'Generating {} training-tasks'.format(n_train_tasks) + '-' * 5, prm.log_file)
    train_tasks_data = [get_data_loader(prm, meta_split='meta_train', limit_train_samples=limit_train_samples) for i_task in range(n_train_tasks)]

    # Meta-training to learn prior:
    prior_model = meta_train_Bayes_finite_tasks.run_meta_learning(train_tasks_data, prm)
    # save learned prior:
    f_path = save_model_state(prior_model, dir_path, name=f_name)
    print('Trained prior saved in ' + f_path)


elif mode == 'LoadPrior':

    # Loads  previously training prior.
    # First, create the model:
    prior_model = get_model(prm, 'Stochastic')
    # Then load the weights:
    load_model_state(prior_model, dir_path, name=f_name)
    print('Pre-trained  prior loaded from ' + dir_path)
else:
    raise ValueError('Invalid mode')

# -------------------------------------------------------------------------------------------
# Generate the data sets of the test tasks:
# -------------------------------------------------------------------------------------------

n_test_tasks = 10
limit_train_samples = 100

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
for i_task in range(n_test_tasks):
    print('Standard learning task {} out of {}...'.format(i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err_standard[i_task], _ = learn_single_standard.run_learning(task_data, prm, verbose=0)


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
run_prior_analysis(prior_model, showPlt=False)


# -------------------------------------------------------------------------------------------
#  Run sequential learning, initialized by learned prior
# -------------------------------------------------------------------------------------------


n_tasks = 200
limit_train_samples = 100

test_err_per_task= np.zeros(n_tasks)

for i_task in range(n_tasks):

    write_result('-'*5 + 'Learning task #{} out of {}...'.format(1+i_task, n_tasks), prm.log_file)
    task_data = get_data_loader(prm, limit_train_samples=limit_train_samples)
    test_err, posterior_model = learn_single_Bayes.run_learning(task_data, prm, prior_model=prior_model, init_from_prior=init_from_prior, verbose=0)
    prior_model = deepcopy(posterior_model)
    test_err_per_task[i_task] = test_err
    write_result('-' * 5 + ' Task {}, test error: {}'.format(1+i_task, test_err), prm.log_file)


# Figure
plt.figure()
plt.plot(1+np.arange(n_tasks),  100 * test_err_per_task)
plt.xlabel('Task')
plt.ylabel('Test Error %')
plt.title('PAC-Bayes Sequential Transfer')
plt.savefig('Figure.png')

run_prior_analysis(prior_model, layers_names=None)



# -------------------------------------------------------------------------------------------
#  Print prior analysis
# -------------------------------------------------------------------------------------------
run_prior_analysis(prior_model)

plt.show()