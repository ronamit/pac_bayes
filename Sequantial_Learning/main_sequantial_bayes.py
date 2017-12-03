from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from Models.models import get_model

from Single_Task import learn_single_Bayes, learn_single_standard
from Utils.data_gen import get_data_loader
from Utils.common import  write_result, set_random_seed
from Stochsastic_Meta_Learning.Analyze_Prior import run_prior_analysis

torch.backends.cudnn.benchmark=True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7


# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'Sinusoid' ",
                    default='MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation: 'None' / 'Permute_Pixels' / 'Permute_Labels'",
                    default='Permute_Pixels')

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

prm.data_path = '../data'

set_random_seed(prm.seed)

#  Define model:
prm.model_name = 'FcNet3'   # 'FcNet2' / 'FcNet3' / 'ConvNet' / 'ConvNet_Dropout'

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr}
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}
# Learning rate decay schedule:
prm.lr_schedule = {} # No decay

# Stochastic learning parameters -
# Weights initialization:
prm.bayes_inits = {'Bayes-Mu': {'bias': 0, 'std': 0.1}, 'Bayes-log-var': {'bias': -10, 'std': 1.0}}
prm.n_MC = 3 # Number of Monte-Carlo iterations
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote'

prm.complexity_type = 'PAC_Bayes_McAllaster'
#  'Variational_Bayes' / 'PAC_Bayes_McAllaster' / 'PAC_Bayes_Pentina' / 'PAC_Bayes_Seeger'  / 'KLD' / 'NoComplexity'

init_from_prior = True  #  False \ True . init posterior from prior

# Create initial prior:
# prior_model = get_model(prm, 'Stochastic')
prior_model = None # Start with no prior

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

plt.show()


