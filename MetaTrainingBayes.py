
from __future__ import absolute_import, division, print_function

import timeit
import data_gen

import numpy as np
import torch
import random
import common as cmn
from common import count_correct, get_param_from_model, grad_step, net_L1_norm
from models_Bayes import get_bayes_model
from meta_utils import get_posterior_complexity_term, get_eps_std

# -------------------------------------------------------------------------------------------
#  Learning function
# -------------------------------------------------------------------------------------------
def run_meta_learning(train_tasks_data, prm, model_type, optim_func, optim_args, loss_criterion, lr_schedule,):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    n_tasks = len(train_tasks_data)

    # Create posterior models for each task:
    posteriors_models = [get_bayes_model(model_type, prm) for _ in range(n_tasks)]

    # Create a 'dummy' model to generate the set of parameters of the shared prior:
    prior_model = get_bayes_model(model_type, prm)

    # number of batches from each task:
    n_batch_list = [len(data_loader['train']) for data_loader in train_tasks_data]

    n_meta_batches = np.min(n_batch_list)

    # Create an optimizer for each tasks' posterior params:
    all_post_param = []
    for i_task in range(n_tasks):
        post_params = list(posteriors_models[i_task].parameters())
        all_post_param += post_params

    # Create optimizer for all parameters (posteriors + prior)
    prior_params = list(prior_model.parameters())
    all_params = all_post_param + prior_params
    all_optimizer = optim_func(all_params, **optim_args)

    # number of training samples in each task :
    n_samples_list = [data_loader['n_train_samples'] for data_loader in train_tasks_data]



    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------
    def run_train_epoch(i_epoch):

        # For each task, prepare an iterator to generate training batches:
        task_train_loaders = [iter(train_tasks_data[i_task]['train']) for i_task in range(n_tasks)]

        for i_batch in range(n_meta_batches):

            eps_std = get_eps_std(i_epoch, i_batch, n_meta_batches, prm)

            sum_empirical_loss = 0
            sum_intra_task_comp = 0

            # In each meta-step, we draws batches from all tasks to calculate the total empirical loss estimate:
            for i_task in range(n_tasks):
                # get data from current task to calculate the empirical loss estimate:
                batch_data = task_train_loaders[i_task].next()

                # The posterior model corresponding to the task in the batch:
                post_model = posteriors_models[i_task]
                post_model.train()

                # get batch variables:
                inputs, targets = data_gen.get_batch_vars(batch_data, prm)

                # Empirical Loss on current task:
                outputs = post_model(inputs, eps_std)
                task_empirical_loss = loss_criterion(outputs, targets)

                # Intra-task complexity of current task:
                task_complexity = get_posterior_complexity_term(
                    prm.complexity_type, prior_model, post_model,
                    n_samples_list[i_task])

                sum_empirical_loss += task_empirical_loss
                sum_intra_task_comp += task_complexity

            # end tasks loop

            # Hyper-prior term:
            hyperprior = net_L1_norm(prior_model)
            hyperprior *= np.sqrt(1 / n_tasks) * prm.hyper_prior_factor

            # Total objective:
            total_objective = (1 / n_tasks) * (sum_empirical_loss + sum_intra_task_comp) + hyperprior

            # Take gradient step with the shared prior and all tasks' posteriors:
            grad_step(total_objective, all_optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            log_interval = 500
            if i_batch % log_interval == 0:
                batch_acc = count_correct(outputs, targets) / prm.batch_size
                print(cmn.status_string(i_epoch, i_batch, n_meta_batches, prm, batch_acc, total_objective.data[0]) +
                      'Eps-STD: {:.4}\tAvg Empiric Loss: {:.4}\t Avg Intra-Comp. {:.4}\t Hyperprior: {:.4}'.
                      format(eps_std, sum_empirical_loss.data[0]/ n_tasks, sum_intra_task_comp.data[0]/ n_tasks, hyperprior.data[0]))
        # end batches loop
    # end run_epoch()

    # -------------------------------------------------------------------------------------------
    #  Test evaluation function -
    # Evaluate the mean loss on samples from the test sets of the training tasks
    # --------------------------------------------------------------------------------------------
    def run_test():
        test_acc_list = []

        for i_task in range(n_tasks):
            model = posteriors_models[i_task]
            test_loader = train_tasks_data[i_task]['test']
            model.eval()
            test_loss = 0
            n_correct = 0
            for batch_data in test_loader:
                inputs, targets = data_gen.get_batch_vars(batch_data, prm)
                eps_std = 0.0  # test with max-posterior
                outputs = model(inputs, eps_std)
                test_loss += loss_criterion(outputs, targets)  # sum the mean loss in batch
                n_correct += count_correct(outputs, targets)

            n_test_samples = len(test_loader.dataset)
            n_test_batches = len(test_loader)
            test_loss = test_loss.data[0] / n_test_batches
            test_acc = n_correct / n_test_samples
            print('Task {}, Test set: Average loss: {:.4}, Accuracy: {:.3} ( {}/{})\n'.format(
                i_task, test_loss, test_acc, n_correct, n_test_samples))
            test_acc_list.append(test_acc)

        return test_acc_list

    # -----------------------------------------------------------------------------------------------------------#
    # Main script
    # -----------------------------------------------------------------------------------------------------------#

    # Update Log file
    run_name = cmn.gen_run_name('Meta-Training')
    cmn.write_result('-'*10+run_name+'-'*10, prm.log_file)
    cmn.write_result(str(prm), prm.log_file)
    cmn.write_result(cmn.get_model_string(prior_model), prm.log_file)
    cmn.write_result(str(optim_func) + str(optim_args) +  str(lr_schedule), prm.log_file)
    cmn.write_result('---- Meta-Training set: {0} tasks'.format(len(train_tasks_data)), prm.log_file)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    for i_epoch in range(prm.num_epochs):
        run_train_epoch(i_epoch)

    stop_time = timeit.default_timer()

    # Test:
    test_acc = run_test()

    # Update Log file:
    test_acc_mean = sum(test_acc, 0) / n_tasks
    cmn.write_final_result(test_acc_mean, stop_time - start_time, prm.log_file)
    cmn.save_code('CodeBackup', run_name)

    # Return learned prior:
    return prior_model
