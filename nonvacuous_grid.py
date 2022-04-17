import argparse
import os
import torch
import torch.optim as optim
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from Utils import data_gen
from Utils.common import set_random_seed, create_result_dir, save_run_data, write_to_log, ensure_dir, load_saved_vars
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

# parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)',
#                     default='')

parser.add_argument('--gpu_index', type=int,
                    help='The index of GPU device to run on',
                    default=0)

parser.add_argument('--seed', type=int, help='random seed',
                    default=1)

parser.add_argument('--test-batch-size', type=int, help='input batch size for testing (reduce if memory is limited)',
                    default=512)

parser.add_argument('--n_MC_eval', type=int,
                    help='number of monte-carlo runs for expected loss estimation and bound evaluation',
                    default=3)

# ----- Task Parameters ---------------------------------------------#

# parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'CIFAR10' / Omniglot / SmallImageNet / binarized_MNIST",
#                     default='CIFAR10')

parser.add_argument('--data-transform', type=str,
                    help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'/ Shuffled_Pixels ",
                    default='None')

parser.add_argument('--limit_train_samples', type=int,
                    help='Upper limit for the number of training samples (0 = unlimited)',
                    default=0)  # 0

# ----- Algorithm Parameters ---------------------------------------------#

# parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM' / Logistic_binary",
#                     default='CrossEntropy')

parser.add_argument('--model_name', type=str, help="Define model type (hypothesis class)'",
                    default='ConvNet3')  # OmConvNet / 'FcNet3' / 'ConvNet3' / OmConvNet_NoBN

parser.add_argument('--batch_size', type=int, help='input batch size for training',
                    default=512)

parser.add_argument('--num_epochs', type=int, help='number of epochs to train, if 0 - use num_iter',
                    default=0)  # 50
parser.add_argument('--num_iter', type=int, help='number of iterations to run, if num_epochs == 0',
                    default=5000)  # 50

parser.add_argument('--lr', type=float, help='learning rate (initial)',
                    default=2e-3)

# parser.add_argument('--override_eps_std', type=float,
#                     help='For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)',
#                     default=1.0)

# -------------------------------------------------------------------------------------------

prm = parser.parse_args()
prm.device = torch.device("cuda:" + str(prm.gpu_index) if torch.cuda.is_available() else "cpu")

prm.log_var_init = {'mean': -5,
                    'std': 0.1}  # The initial value for the log-var parameter (rho) of each weight of the posterior, in case init_from_prior==False

# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 1

# prm.use_randomness_schedeule = True # False / True
# prm.randomness_init_epoch = 0
# prm.randomness_full_epoch = 500000000

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam, {'lr': prm.lr}
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10, 30]}
prm.lr_schedule = {}  # No decay

# Test type:
prm.test_type = 'Expected'  # 'MaxPosterior' / 'MajorityVote' / 'Expected'

# Learning objective parameters
prm.complexity_type = 'New_PB'  # 'McAllester' / 'Seeger' / 'NoComplexity' / 'New_PB'
prm.divergence_type = 'KL'  # 'KL' / 'W_Sqr' /  'W_NoSqr' /
prm.delta = 0.035  # maximal probability that the bound does not hold

# prm.prior_log_var = -5
# prm.prior_mean = 0
prm.prior_log_var = {'mean': -5, 'std': 0.1}
prm.prior_mean = {'mean': 0, 'std': 0.1}
prm.posterior_init_noise = 1e-2

##-------- Binary-class MNIST --------
run_name = 'BinMNIST'
prm.loss_type = 'Logistic_binary'
prm.data_source = 'binarized_MNIST'
samp_grid_delta = 50
max_grid = 500
loss_type_eval = 'Zero_One'
n_reps = 1


# # ---------Multi-class MNIST  --------
# prm.loss_type = 'CrossEntropy'
# prm.data_source = 'MNIST'    #
# samp_grid_delta = 20
# max_grid = 400
# loss_type_eval = 'Zero_One'
# n_reps = 1
# run_name = f'MultiMNIST_1_reps_up_to_400'

# # # ##---------CIFAR 10 --------
# run_name = 'CIFAR10_5k_grid_20_reps_100_Epochs_NewPrior_NoBN'
# prm.loss_type = 'CrossEntropy'
# prm.data_source = 'CIFAR10'
# samp_grid_delta = 5000
# max_grid = 50000
# loss_type_eval = 'Zero_One'
# n_reps = 20


# Run params:
run_experiments = True  # True/False If false, just analyze the previously saved experiments

# grid parameters:
train_samples_vec = samp_grid_delta * np.arange(1, 1 + np.floor(max_grid / samp_grid_delta)).astype(int)

val_types = [['train_loss'], ['test_loss'],
             ['Bound', 'Classic_PB', 'KL'], ['Bound', 'New_PB', 'KL']]
# ['Divergence', 'KL'], ['Divergence', 'W_Sqr']]
# ['Bound', 'Seeger', 'KL'], ['Bound', 'Seeger', 'W_Sqr'], ['Bound', 'Seeger', 'W_NoSqr'],


file_name = 'run_data.pkl'

prm.run_name = run_name
create_result_dir(prm, run_experiments)
path_to_result_file = os.path.join(prm.result_dir, file_name)

if run_experiments:
    # -------------------------------------------------------------------------------------------
    #  Init run
    # -------------------------------------------------------------------------------------------

    prm.data_path = get_data_path()
    set_random_seed(prm.seed)

    task_generator = data_gen.Task_Generator(prm)

    n_val_types = len(val_types)
    n_grid = len(train_samples_vec)
    val_mat = np.zeros((n_val_types, n_grid, n_reps))

    # -------------------------------------------------------------------------------------------
    #  Run grid
    # -------------------------------------------------------------------------------------------
    prm_eval = deepcopy(prm)  # parameters for evaluation
    prm_eval.loss_type = loss_type_eval
    for i_grid, n_train_samples in enumerate(train_samples_vec):

        for i_rep in range(n_reps):
            # Generate task data set:
            data_loader = task_generator.get_data_loader(prm, limit_train_samples=n_train_samples)

            #  Create Prior
            prior_model = get_model(prm, requires_grad=False)
            set_model_values(prior_model, prm.prior_mean, prm.prior_log_var)

            # Learn a posterior which minimizes some bound with the training loss function
            post_model, test_err, test_loss, log_mat = learn_single_Bayes.run_learning(data_loader, prm, prior_model,
                                                                                       init_from_prior=True)

            # evaluation
            _, train_loss = run_eval_Bayes(post_model, data_loader['train'], prm_eval)
            _, test_loss = run_eval_Bayes(post_model, data_loader['test'], prm_eval)
            write_to_log('n_samples: {}, rep: {}.  Train-loss :{:.4}, Test-loss:  {:.4}'.format(n_train_samples, i_rep,
                                                                                                train_loss, test_loss),
                         prm)

            for i_val_type, val_type in enumerate(val_types):
                if val_type[0] == 'train_loss':
                    val = train_loss
                elif val_type[0] == 'test_loss':
                    val = test_loss
                elif val_type[0] == 'Bound':
                    prm_eval.complexity_type = val_type[1]
                    prm_eval.divergence_type = val_type[2]
                    val = learn_single_Bayes.eval_bound(post_model, prior_model, data_loader, prm_eval, train_loss)
                    write_to_log(str(val_type) + ' = ' + str(val), prm)
                elif val_type[0] == 'Divergence':
                    prm_eval.divergence_type = val_type[1]
                    val = get_net_densities_divergence(prior_model, post_model, prm_eval)
                else:
                    raise ValueError('Invalid val_types')

                val_mat[i_val_type, i_grid, i_rep] = val
            # end val_types loop
        # end reps loop
    # end grid loop

    # Saving the analysis:
    save_run_data(prm, {'val_mat': val_mat, 'loss_type_eval': loss_type_eval, 'train_samples_vec': train_samples_vec,
                        'val_types': val_types})

else:
    loaded_prm, loaded_dict = load_saved_vars(prm.result_dir)
    prm = loaded_prm
    set_random_seed(prm.seed)
    # get learned posterior and results
    val_mat, loss_type_eval, train_samples_vec, val_types = loaded_dict.values()

# end if run_experiments


val_types_for_show = [['train_loss'], ['test_loss'],
                      ['Bound', 'Classic_PB', 'KL'], ['Bound', 'New_PB', 'KL'],
                      ['Bound', 'Seeger', 'KL']]

# val_types_for_show = [['train_loss'], ['test_loss'],
#              ['Bound', 'McAllester', 'W_Sqr'], ['Bound', 'McAllester', 'W_NoSqr'],
#              ['Bound', 'Seeger', 'W_Sqr'], ['Bound', 'Seeger', 'W_NoSqr']]

# val_types_for_show =  [['Divergence', 'KL'], ['Divergence', 'W_Sqr']]

# val_types_for_show =  [['Divergence', 'W_Sqr']]

# Plot the analysis:
plt.figure()
for i_val_type, val_type in enumerate(val_types):
    if val_type in val_types_for_show:
        plt.errorbar(train_samples_vec,
                     val_mat[i_val_type].mean(axis=1),
                     yerr=val_mat[i_val_type].std(axis=1),
                     label=str(val_type))

# plt.xticks(train_samples_vec)
plt.xlabel('Number of Samples')
plt.ylabel(loss_type_eval)
plt.legend()
plt.title(prm.run_name)
# plt.savefig(root_saved_dir + base_run_name+'.pdf', format='pdf', bbox_inches='tight')
# plt.ylim([0, 0.2])
plt.grid()
plt.savefig('Fig.png', format='png', bbox_inches='tight')
plt.show()


