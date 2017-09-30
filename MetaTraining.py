
from __future__ import absolute_import, division, print_function
from six.moves import xrange
import timeit
import data_gen

import numpy as np
import torch
import random
import common as cmn
from common import count_correct, get_param_from_model, grad_step
from models_standard import get_model
from bayes_func import get_intra_task_complexity

# -------------------------------------------------------------------------------------------
#  Learning function
# -------------------------------------------------------------------------------------------
def run_meta_learning(train_tasks_data, prm, model_type, optim_func, optim_args, loss_criterion, lr_schedule, complexity_type):


    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    n_tasks = len(train_tasks_data)

    # Create posterior models for each task:
    posteriors_models = [get_model(model_type, prm) for _ in xrange(n_tasks)]

    # Create a 'dummy' model to generate the set of parameters of the shared prior:
    prior_means_model = get_model(model_type, prm)
    prior_log_vars_model = get_model(model_type, prm)

    # number of batches from each task:
    n_batch_list = [len(data_loader['train']) for data_loader in train_tasks_data]
    # total number of batches in an epoch:
    n_batches = sum(n_batch_list, 0)

    # Create an optimizer for each tasks' posterior params:
    post_optimizers = []
    all_post_param = []
    for i_task in xrange(n_tasks):
        post_params = list(posteriors_models[i_task].parameters())
        all_post_param += post_params
        post_optimizers.append(optim_func(post_params, **optim_args))

    # Create optimizer for all parameters (posteriors + prior)
    prior_params = list(prior_means_model.parameters()) + list(prior_log_vars_model.parameters())
    all_params = all_post_param + prior_params
    all_optimizer = optim_func(all_params, **optim_args)

    param_names_list = [param_name for param_name, param  in prior_means_model.named_parameters()]
    n_param = len(param_names_list)

    # number of training samples in each task :
    n_samples_list = [data_loader['n_train_samples'] for data_loader in train_tasks_data]

    # Generate a list of the task index to use in each training batch
    # (ther list order will be shuffled in each epoch start)
    task_order = sum([[i_task] * n_batch_list[i_task] for i_task in xrange(n_tasks)], [])

    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------
    def run_train_epoch(i_epoch):

        # For each task, prepare an iterator to generate training batches:
        task_train_loaders = [iter(train_tasks_data[i_task]['train']) for i_task in xrange(n_tasks)]

        # Shuffle task order for batch generation:
        random.shuffle(task_order)

        for batch_idx, task_id in enumerate(task_order):

            # get data drom current task to calculate the empirical loss estimate:
            batch_data = task_train_loaders[task_id].next()

            # The posterior model corresponding to the task in the batch:
            post_model = posteriors_models[task_id]
            post_model.train()

            # get batch variables:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)

            # Empirical Loss:
            outputs = post_model(inputs)
            empirical_loss = loss_criterion(outputs, targets)

            # Calculate the average intra-task complexity (for all tasks)  and update the prior,
            # each 'prior_update_interval' steps.
            if batch_idx % prm.prior_update_interval == 0:

                avg_intra_task_comp = 0
                for i_task in range(n_tasks):
                    avg_intra_task_comp += get_intra_task_complexity(
                        complexity_type, prior_means_model, prior_log_vars_model, posteriors_models[i_task], n_samples_list[i_task])
                avg_intra_task_comp /= n_tasks

                # Total objective:
                hyper_prior_factor = 1e-6 * np.sqrt(1 / n_tasks)
                hyper_prior = 0 * hyper_prior_factor # TODO: add hyper_prior
                total_objective = empirical_loss + avg_intra_task_comp + hyper_prior

                # Take gradient step with the shared prior and all tasks' posteriors:
                grad_step(total_objective, all_optimizer, lr_schedule, prm.lr, i_epoch)

            else:
                # Update only the current task posterior, based on its empirical loss and the the its intra-task complexity
                intra_task_comp =  get_intra_task_complexity(
                        complexity_type, prior_means_model, prior_log_vars_model, posteriors_models[task_id], n_samples_list[task_id])
                total_objective = empirical_loss + intra_task_comp
                optimizer = post_optimizers[task_id]
                grad_step(total_objective, optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            log_interval = 500
            if batch_idx % log_interval == 0:
                batch_acc = count_correct(outputs, targets) / prm.batch_size
                print(cmn.status_string(i_epoch, batch_idx, n_batches, prm, batch_acc, total_objective.data[0]))

        # end batch loop
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
                outputs = model(inputs)
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
    # Update Log file
    # -----------------------------------------------------------------------------------------------------------#

    run_name = cmn.gen_run_name('Meta-Training')
    cmn.write_result('-'*10+run_name+'-'*10, prm.log_file)
    cmn.write_result(str(prm), prm.log_file)
    cmn.write_result(cmn.get_model_string(prior_means_model), prm.log_file)
    cmn.write_result(str(optim_func) + str(optim_args) +  str(lr_schedule), prm.log_file)
    cmn.write_result('---- Meta-Training set: {0} tasks'.format(len(train_tasks_data)), prm.log_file)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    startRuntime = timeit.default_timer()

    # Training loop:
    for i_epoch in range(prm.num_epochs):
        run_train_epoch(i_epoch)

    stopRuntime = timeit.default_timer()

    # Test:
    test_acc = run_test()

    # Update Log file:
    test_acc_mean = sum(test_acc, 0) / n_tasks
    cmn.write_final_result(test_acc_mean, stopRuntime - startRuntime, prm.log_file)
    cmn.save_code('CodeBackup', run_name)

    # Return learned prior:
    prior_dict = {'means_model': prior_means_model, 'log_var_model': prior_log_vars_model}
    return prior_dict
