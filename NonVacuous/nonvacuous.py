
from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.optim as optim
from copy import deepcopy

from Utils import data_gen
from Utils.common import set_random_seed, create_result_dir, save_run_data, write_to_log
from Single_Task import learn_single_Bayes, learn_single_standard
from Data_Path import get_data_path
from Models.stochastic_models import get_model
from Utils.Bayes_utils import set_model_values, run_eval_Bayes
from Utils.complexity_terms import get_net_densities_divergence
from Utils.data_gen import get_info

torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'CIFAR10' / Omniglot / SmallImageNet / binarized_MNIST",
                    default='binarized_MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'/ Shuffled_Pixels ",
                    default='None')

parser.add_argument('--limit_train_samples', type=int,
                    help='Upper limit for the number of training samples (0 = unlimited)',
                    default=0)

# ----- Algorithm Parameters ---------------------------------------------#

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM' / Logistic_binary",
                    default='Logistic_binary')

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='FcNet3')  # OmConvNet / 'FcNet3' / 'ConvNet3'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=50)  # 50

parser.add_argument('--lr', type=float, help='learning rate (initial)',
                    default=1e-3)

# parser.add_argument('--override_eps_std', type=float,
#                     help='For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)',
#                     default=1.0)

# -------------------------------------------------------------------------------------------

prm = parser.parse_args()


prm.log_var_init = {'mean': -5, 'std': 0.1} # The initial value for the log-var parameter (rho) of each weight

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
prm.lr_schedule = {}  # No decay

# Test type:
prm.test_type = 'Expected' # 'MaxPosterior' / 'MajorityVote' / 'Expected'
prm.n_MC_eval = 10 # number of monte-carlo runs for expected loss estimation and bound evaluation

# Learning objective parameters
prm.complexity_type = 'McAllaster'  # 'McAllaster' / 'Seeger'
prm.divergence_type = 'W_Sqr'    # 'KL' / 'W_Sqr' /  'W_NoSqr'
prm.delta = 0.035   # maximal probability that the bound does not hold

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

# create model of the network
prior_model = get_model(prm)
set_model_values(prior_model, prior_mean, prior_log_var)

#### DEBUG #########################3
# import math
# prt = deepcopy(prm) #  temp parameters
# prt.divergence_type = 'W_Sqr'
# model1 = get_model(prm)
# m1 = 0.0
# m2 = 0.0
# s1 = 2.0
# s2 = 5.0
# d = model1.weights_count
# set_model_values(model1, mean=m1, log_var=2*math.log(s1))
# model2 = get_model(prm)
# set_model_values(model2, mean=m2, log_var=2*math.log(s2))
# div_val = get_net_densities_divergence(model1, model2, prt)
# print('DEBUG: Divergence value computed {}'.format(div_val))
# print('DEBUG: Divergence value analytic {}'.format(d*((m1-m2)**2 + (s1-s2)**2)))
####################

# -------------------------------------------------------------------------------------------
#  Run learning
# -------------------------------------------------------------------------------------------
# Learn a posterior which minimizes some bound with some loss function
post_model, test_err, test_loss = learn_single_Bayes.run_learning(data_loader, prm, prior_model, init_from_prior=True)

save_run_data(prm, {'test_err': test_err, 'test_loss': test_loss})


# -------------------------------------------------------------------------------------------
#  Evaluate bounds
# -------------------------------------------------------------------------------------------
# Calculate bounds the expected risk of the learned posterior
# Note: the bounds are evaluated using only the training data
# But they should upper bound the test-loss (with high probability)


# Choose the appropriate loss functions for evaluation:
info = get_info(prm)
if info['type'] == 'multi_class':
    losses = ['Zero_One_Multi']
elif info['type'] == 'binary_class':
    losses = ['Logistic_Binary_Clipped', 'Zero_One_Binary']
else:
    raise ValueError

prt = deepcopy(prm) #  temp parameters
for loss_type in losses:
    prt.loss_type = loss_type
    test_acc, test_loss = run_eval_Bayes(post_model, data_loader['test'], prt)
    train_acc, train_loss = run_eval_Bayes(post_model, data_loader['train'], prt)
    print('-'*20)
    write_to_log('-Loss func. {}, Train-loss :{:.4}, Test-loss:  {:.4}'.format(loss_type, train_loss, test_loss), prm)

    for divergence_type in ['KL', 'W_Sqr', 'W_NoSqr']:
        prt.divergence_type = divergence_type
        div_val = get_net_densities_divergence(prior_model, post_model, prt)
        write_to_log('\t--Divergence: {} = {:.4}'.format(prt.divergence_type, div_val), prm)
        for complexity_type in ['McAllaster', 'Seeger', 'Catoni']:
            prt.complexity_type = complexity_type
            bound_val = learn_single_Bayes.eval_bound(post_model, prior_model, data_loader, prt)
            write_to_log('\t\tBound: {} =  {:.4}'.
                         format(prt.complexity_type, bound_val), prm)

# -------------------------------------------------------------------------------------------
#  Run standard deterministic learning for comparision
# -------------------------------------------------------------------------------------------
# test_err, test_loss = learn_single_standard.run_learning(data_loader, prm)


