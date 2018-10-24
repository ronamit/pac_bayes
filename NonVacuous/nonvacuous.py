
from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.optim as optim
from Utils import data_gen
from Utils.common import set_random_seed, create_result_dir, save_run_data
from Single_Task import learn_single_Bayes
from Data_Path import get_data_path

torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# ----- Run Parameters ---------------------------------------------#

parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)',
                    default='')

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing (reduce if memory is limited)',
                    default=128)

# ----- Task Parameters ---------------------------------------------#

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'CIFAR10' / Omniglot / SmallImageNet",
                    default='binarized_MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'/ Shuffled_Pixels ",
                    default='None')

parser.add_argument('--limit_train_samples', type=int,
                    help='Upper limit for the number of training samples (0 = unlimited)',
                    default=0)

# ----- Algorithm Parameters ---------------------------------------------#

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='ConvNet3')  # OmConvNet / 'FcNet3' / 'ConvNet3'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=50) # 300

parser.add_argument('--lr', type=float, help='learning rate (initial)',
                    default=1e-3)

# parser.add_argument('--override_eps_std', type=float,
#                     help='For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)',
#                     default=1.0)

# -------------------------------------------------------------------------------------------

prm = parser.parse_args()


prm.log_var_init = {'mean': -10, 'std': 0.1} # The initial value for the log-var parameter (rho) of each weight

# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 1

# prm.use_randomness_schedeule = True # False / True
# prm.randomness_init_epoch = 0
# prm.randomness_full_epoch = 500000000

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr}
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10, 30]}
prm.lr_schedule = {} # No decay

# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote'


# Bound parameters
prm.complexity_type = 'NewBoundMcAllaster'
prm.divergence_type = 'Wasserstein'    # 'KL' / 'Wasserstein' /  'Wasserstein_NoSqrt'
prm.delta = 0.1  #  maximal probability that the bound does not hold

# -------------------------------------------------------------------------------------------
#  Init run
# -------------------------------------------------------------------------------------------

prm.data_path = get_data_path()
set_random_seed(prm.seed)
create_result_dir(prm)

# Generate task data set:
task_generator = data_gen.Task_Generator(prm)
data_loader = task_generator.get_data_loader(prm, limit_train_samples=prm.limit_train_samples)



# -------------------------------------------------------------------------------------------
#  Create Prior
# -------------------------------------------------------------------------------------------

prior_log_var = -5
prior_mean = 0

from Models.stochastic_models import get_model
from Models.stochastic_layers import StochasticLayer


# create model of the network (with random init)
prior_model = get_model(prm)

layers_list = [layer for layer in prior_model.children() if isinstance(layer, StochasticLayer)]

for i_layer, layer in enumerate(layers_list):
    if hasattr(layer, 'w'):
        layer.w['log_var'].data.fill_(prior_log_var)
        layer.w['mean'].data.fill_(prior_mean)
    if hasattr(layer, 'b'):
        layer.b['log_var'].data.fill_(prior_log_var)
        layer.b['mean'].data.fill_(prior_mean)




# -------------------------------------------------------------------------------------------
#  Run learning
# -------------------------------------------------------------------------------------------

test_err, _ = learn_single_Bayes.run_learning(data_loader, prm, prior_model, init_from_prior=True)

save_run_data(prm, {'test_err': test_err})