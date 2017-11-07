from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from Models.models import get_model

from Single_Task import learn_single_Bayes, learn_single_standard
from Utils import data_gen
from Utils.common import  write_result, set_random_seed


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

# Weights initialization:
prm.init_override = None # None = use default initializer
# prm.init_override = {'mean': 0, 'std': 0.1}

# Create initial prior:
prev_model = get_model(prm, 'Standard', prm.init_override)


n_tasks = 50
limit_train_samples = 100

test_err_per_task= np.zeros(n_tasks)

for i_task in range(n_tasks):

    write_result('-'*5 + 'Learning task #{} out of {}...'.format(i_task, n_tasks), prm.log_file)
    task_data = data_gen.get_data_loader(prm, limit_train_samples=limit_train_samples)
    test_err, new_model = learn_single_standard.run_learning(task_data, prm, initial_model=prev_model, verbose=0)
    prev_model = deepcopy(new_model)
    test_err_per_task[i_task] = test_err
    write_result('-' * 5 + ' Task {}, test error: '.format(test_err), prm.log_file)


# Figure
plt.figure()
plt.plot(1+np.arange(n_tasks),  100 * test_err_per_task)
plt.xlabel('Task')
plt.ylabel('Test Error %')
plt.title('Warm-Start Sequential Transfer')
plt.show()
plt.savefig('Figure.png')




