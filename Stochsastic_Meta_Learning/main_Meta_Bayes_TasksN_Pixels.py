
from __future__ import absolute_import, division, print_function

import argparse
import timeit, time
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim

from Stochsastic_Meta_Learning import meta_test_Bayes, meta_train_Bayes_finite_tasks, meta_train_Bayes_infinite_tasks
from Single_Task import learn_single_Bayes
from Models.stochastic_models import get_model
from Single_Task import learn_single_standard
from Utils.data_gen import Task_Generator
from Utils.common import save_model_state, load_model_state, write_result, set_random_seed

torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs
# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()


parser.add_argument('--log-file', type=str, help='Name of file to save log (None = no save)',
                    default='log')
parser.add_argument('--n_pixels_shuffels', type=int, help='For "Shuffled_Pixels": how many pixels swaps',
                    default=300)



parser.add_argument('--data-source', type=str, help='Data set',
                    default='MNIST') # 'MNIST' / 'Omniglot'

parser.add_argument('--max_n_tasks', type=int, help='Number of meta-training tasks (0 = infinite)',
                    default=10)

parser.add_argument('--data-transform', type=str, help="Data transformation",
                    default='Shuffled_Pixels') #  'None' / 'Permute_Pixels' / 'Permute_Labels' / Rotate90 Shuffled_Pixels

parser.add_argument('--loss-type', type=str, help="Loss function",
                    default='CrossEntropy') #  'CrossEntropy' / 'L2_SVM'

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='FcNet3')  # ConvNet3 / 'FcNet3'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--n_meta_train_epochs', type=int, help='number of epochs to train',
                    default=300)  # 10 / 100
parser.add_argument('--n_inner_steps', type=int, help='For infinite tasks case, number of steps for training per meta-batch of tasks',
                    default=50)  #

parser.add_argument('--n_meta_test_epochs', type=int, help='number of epochs to train',
                    default=300)  # 10 / 300

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=1000)

parser.add_argument('--meta_batch_size', type=int, help='Maximal number of tasks in each meta-batch',
                    default=16)
# Run parameters:
parser.add_argument('--mode', type=str, help='MetaTrain or LoadMetaModel',
                    default='MetaTrain')   # 'MetaTrain'  \ 'LoadMetaModel'

parser.add_argument('--meta_model_file_name', type=str, help='File name to save meta-model or to load from',
                    default='meta_model')

parser.add_argument('--limit_train_samples_in_test_tasks', type=int,
                    help='Upper limit for the number of training sampels in the meta-test tasks (0 = unlimited)',
                    default=2000)

parser.add_argument('--n_test_tasks', type=int,
                    help='Number of meta-test tasks for meta-evaluation',
                    default=10)



# Omniglot Parameters:
parser.add_argument('--N_Way', type=int, help='Number of classes in a task (for Omniglot)',
                    default=5)
parser.add_argument('--K_Shot', type=int, help='Number of training sample per class (for Omniglot)',
                    default=5)  # Note: number of test samples per class is 20-K (the rest of the data)
parser.add_argument('--chars_split_type', type=str, help='how to split the Omniglot characters  - "random" / "predefined_split"',
                    default='random')
parser.add_argument('--n_meta_train_chars', type=int, help='For Omniglot: how many characters to use for meta-training, if split type is random',
                    default=1200)

parser.add_argument('--override_eps_std', type=float,
                    help='For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)',
                    default=1.0)


parser.add_argument('--complexity_type', type=str,
                    help=" 'Variational_Bayes' / 'PAC_Bayes_McAllaster' / 'PAC_Bayes_Pentina' / 'PAC_Bayes_Seeger'  / 'KLD' / 'NoComplexity' /  NewBoundMcAllaster / NewBoundSeeger'",
                    default='NewBoundSeeger')



prm = parser.parse_args()

from Data_Path import get_data_path
prm.data_path = get_data_path()

set_random_seed(prm.seed)

prm.num_epochs = prm.n_meta_test_epochs

# Weights initialization (for Bayesian net):
prm.log_var_init = {'mean':-10, 'std':0.1} # The initial value for the log-var parameter (rho) of each weight

# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 1

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr} #'weight_decay': 1e-4
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [50, 150]}
prm.lr_schedule = {}  # No decay

# MPB alg  params:
prm.kappa_prior = 2e3  #  parameter of the hyper-prior regularization
prm.kappa_post = 1e-3  # The STD of the 'noise' added to prior
prm.delta = 0.1  #  maximal probability that the bound does not hold

init_from_prior = True  #  False \ True . In meta-testing -  init posterior from learned prior

# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote' / 'AvgVote'

dir_path = './saved'

task_generator = Task_Generator(prm)
# -------------------------------------------------------------------------------------------
#  Run Meta-Training
# -------------------------------------------------------------------------------------------

start_time = timeit.default_timer()

max_n_tasks = prm.max_n_tasks
n_tasks_vec = np.arange(1, max_n_tasks+1)
mean_error_per_tasks_n = np.zeros(len(n_tasks_vec))
std_error_per_tasks_n = np.zeros(len(n_tasks_vec))



for i_task_n, n_train_tasks in enumerate(n_tasks_vec):

    if n_train_tasks:
        # In this case we generate a finite set of train (observed) task before meta-training.
        # Generate the data sets of the training tasks:
        write_result('-' * 5 + 'Generating {} training-tasks'.format(n_train_tasks) + '-' * 5, prm.log_file)
        train_data_loaders = task_generator.create_meta_batch(prm, n_train_tasks, meta_split='meta_train')

        # Meta-training to learn prior:
        prior_model = meta_train_Bayes_finite_tasks.run_meta_learning(train_data_loaders, prm)
        # save learned prior:
        f_path = save_model_state(prior_model, dir_path, name=prm.meta_model_file_name)
        print('Trained prior saved in ' + f_path)
    else:
       # learn from scratch
       prior_model = None

    # -------------------------------------------------------------------------------------------
    # Generate the data sets of the test tasks:
    # -------------------------------------------------------------------------------------------

    n_test_tasks = prm.n_test_tasks

    limit_train_samples_in_test_tasks = prm.limit_train_samples_in_test_tasks
    if limit_train_samples_in_test_tasks == 0:
        limit_train_samples_in_test_tasks = None

    write_result('-'*5 + 'Generating {} test-tasks with at most {} training samples'.
                 format(n_test_tasks, limit_train_samples_in_test_tasks)+'-'*5, prm.log_file)


    test_tasks_data = task_generator.create_meta_batch(prm, n_test_tasks, meta_split='meta_test',
                                                           limit_train_samples=limit_train_samples_in_test_tasks)
    #
    # -------------------------------------------------------------------------------------------
    #  Run Meta-Testing
    # -------------------------------------------------------------------------------
    write_result('Meta-Testing with transferred prior....', prm.log_file)

    test_err_bayes = np.zeros(n_test_tasks)
    for i_task in range(n_test_tasks):
        print('Meta-Testing task {} out of {}...'.format(1+i_task, n_test_tasks))
        task_data = test_tasks_data[i_task]
        if prior_model:
            test_err_bayes[i_task], _ = meta_test_Bayes.run_learning(task_data, prior_model, prm, init_from_prior, verbose=0)
        else:
            # learn from scratch
            test_err_bayes[i_task], _ = learn_single_Bayes.run_learning(task_data, prm, verbose=0)

    mean_error_per_tasks_n[i_task_n]  = test_err_bayes.mean()
    std_error_per_tasks_n[i_task_n] = test_err_bayes.std()
# end n_tasks loop



# -------------------------------------------------------------------------------------------
#  Print results
# -------------------------------------------------------------------------------------------


import pickle, os
# Saving the objects:
results_file_name = prm.log_file
with open(os.path.join(dir_path, results_file_name+'.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([mean_error_per_tasks_n, std_error_per_tasks_n, n_tasks_vec], f)

# # Getting back the objects:
# with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
#     obj0, obj1, obj2 = pickle.load(f)


import matplotlib.pyplot as plt
plt.figure()

plt.errorbar(n_tasks_vec, 100*mean_error_per_tasks_n, yerr=100*std_error_per_tasks_n)
plt.xticks(n_tasks_vec)
plt.xlabel('Number of training-tasks')
plt.ylabel('Error on new task [%]')
plt.show()

stop_time = timeit.default_timer()
write_result('Total runtime: ' +
             time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)),  prm.log_file)
# -------------------------------------------------------------------------------------------
