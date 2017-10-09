
from __future__ import absolute_import, division, print_function

import timeit, copy

import numpy as np

from Models.models_standard import get_model
from Utils import common as cmn, data_gen
from Utils.common import count_correct, grad_step, net_L1_norm, correct_rate, get_loss_criterion, write_result
from Deterministic_Meta_Learning.utils_meta_det import get_weights_complexity_term


# -------------------------------------------------------------------------------------------
#  Learning function
# -------------------------------------------------------------------------------------------
def run_meta_learning(train_tasks_data, prm, model_type):


    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------

    # Unpack parameters:
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)


    n_tasks = len(train_tasks_data)

    # Create posterior models for each task:
    posteriors_models = [get_model(model_type, prm) for _ in range(n_tasks)]

    # Create a 'dummy' model to generate the set of parameters of the shared prior:
    prior_means_model = get_model(model_type, prm)
    log_var_model_prm = copy.copy(prm)
    log_var_model_prm.weights_init_bias = -10  # set the initial sigma to a low value
    prior_log_vars_model = get_model(model_type, log_var_model_prm)

    # number of batches from each task:
    n_batch_list = [len(data_loader['train']) for data_loader in train_tasks_data]

    n_meta_batches = np.min(n_batch_list)

    # Create an optimizer for each tasks' posterior params:
    all_post_param = []
    for i_task in range(n_tasks):
        post_params = list(posteriors_models[i_task].parameters())
        all_post_param += post_params

    # Create optimizer for all parameters (posteriors + prior)
    prior_params = list(prior_means_model.parameters()) + list(prior_log_vars_model.parameters())
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
                outputs = post_model(inputs)
                task_empirical_loss = loss_criterion(outputs, targets)

                # Intra-task complexity of current task:
                task_complexity = get_weights_complexity_term(
                    prm.complexity_type, prior_means_model, prior_log_vars_model, posteriors_models[i_task],
                    n_samples_list[i_task])

                sum_empirical_loss += task_empirical_loss
                sum_intra_task_comp += task_complexity

            # end tasks loop

            # Hyper-prior term:
            hyperprior = net_L1_norm(prior_log_vars_model) + net_L1_norm(prior_log_vars_model)
            hyperprior *= np.sqrt(1 / n_tasks) * prm.hyper_prior_factor

            # Total objective:
            total_objective = (1 / n_tasks) * (sum_empirical_loss + sum_intra_task_comp) + hyperprior

            # Take gradient step with the shared prior and all tasks' posteriors:
            grad_step(total_objective, all_optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            log_interval = 500
            if i_batch % log_interval == 0:
                batch_acc = correct_rate(outputs, targets)
                print(cmn.status_string(i_epoch, i_batch, n_meta_batches, prm, batch_acc, total_objective.data[0]))
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
    write_result('-'*10+run_name+'-'*10, prm.log_file)
    write_result(str(prm), prm.log_file)
    write_result(cmn.get_model_string(prior_means_model), prm.log_file)
    write_result('Total number of steps: {}'.format(n_meta_batches * prm.num_epochs), prm.log_file)
    write_result('---- Meta-Training set: {0} tasks'.format(len(train_tasks_data)), prm.log_file)

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


    # Return learned prior:
    prior_dict = {'means_model': prior_means_model, 'log_var_model': prior_log_vars_model}
    return prior_dict
