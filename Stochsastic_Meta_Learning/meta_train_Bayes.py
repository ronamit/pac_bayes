
from __future__ import absolute_import, division, print_function

import timeit
import random
import numpy as np

from Models.models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import get_posterior_complexity_term, get_eps_std, run_test_Bayes
from Utils.common import grad_step, net_L1_norm, correct_rate, get_loss_criterion, write_result


# -------------------------------------------------------------------------------------------
#  Learning function
# -------------------------------------------------------------------------------------------
def run_meta_learning(train_tasks_data, prm):

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
    posteriors_models = [get_model(prm, 'Stochastic') for _ in range(n_tasks)]

    # Create a 'dummy' model to generate the set of parameters of the shared prior:
    prior_model = get_model(prm, 'Stochastic')

    # Gather all tasks posterior params:
    all_post_param = []
    for i_task in range(n_tasks):
        post_params = list(posteriors_models[i_task].parameters())
        all_post_param += post_params

    # Create optimizer for all parameters (posteriors + prior)
    prior_params = list(prior_model.parameters())
    all_params = all_post_param + prior_params
    all_optimizer = optim_func(all_params, **optim_args)

    # Create optimizer for only the posteriors
    posteriors_optimizer = optim_func(all_post_param, **optim_args)

    # number of training samples in each task :
    n_samples_list = [ data_loader['n_train_samples'] for data_loader in train_tasks_data]

    # number of sample-batches in each task:
    n_batch_list = [len(data_loader['train']) for data_loader in train_tasks_data]

    n_batches_per_task = np.min(n_batch_list)
    # note: if some tasks have less data that other tasks - it may be sampled more than once in an epoch


    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------
    def run_train_epoch(i_epoch):

        # For each task, prepare an iterator to generate training batches:
        task_train_loaders = [iter(train_tasks_data[ii]['train']) for ii in range(n_tasks)]

        # The task order to take batches from:
        task_order = []
        task_ids_list = list(range(n_tasks))
        for i_batch in range(n_batches_per_task):
            random.shuffle(task_ids_list)
            task_order += task_ids_list

        # meta-batches loop
        # each meta-batch includes summing over several tasks batches
        # take a grad step after each meta-batch
        meta_batch_starts = list(range(0, len(task_order), prm.meta_batch_size))
        n_meta_batches = len(meta_batch_starts)

        for i_meta_batch in range(n_meta_batches):

            eps_std = get_eps_std(i_epoch, i_meta_batch, n_meta_batches, prm)

            meta_batch_start = meta_batch_starts[i_meta_batch]
            task_ids_in_meta_batch = task_order[meta_batch_start: (meta_batch_start + prm.meta_batch_size)]
            n_inner_batch = len(task_ids_in_meta_batch)  # it may be less than  prm.meta_batch_size at the last one
            # note: it is OK if some task appear several times in the meta-batch

            sum_empirical_loss = 0
            sum_intra_task_comp = 0

            # samples-batches loop (inner loop)
            for i_inner_batch in range(n_inner_batch):

                task_id = task_ids_in_meta_batch[i_inner_batch]

                # get sample-batch data from current task to calculate the empirical loss estimate:
                try:
                    batch_data = task_train_loaders[task_id].next()
                except StopIteration:
                    # in case some task has less samples - just restart the iterator and re-use the samples
                    task_train_loaders[task_id] = iter(train_tasks_data[task_id]['train'])
                    batch_data = task_train_loaders[task_id].next()

                # The posterior model corresponding to the task in the batch:
                post_model = posteriors_models[task_id]
                post_model.train()

                # Monte-Carlo iterations:
                n_MC = prm.n_MC if eps_std > 0 else 1
                task_empirical_loss = 0
                for i_MC in range(n_MC):
                # get batch variables:
                    inputs, targets = data_gen.get_batch_vars(batch_data, prm)

                    # Empirical Loss on current task:
                    outputs = post_model(inputs, eps_std)
                    task_empirical_loss += (1 / n_MC) * loss_criterion(outputs, targets)
                # end MC loop

                # Intra-task complexity of current task:
                task_complexity = get_posterior_complexity_term(
                    prm, prior_model, post_model,
                    n_samples_list[task_id], task_empirical_loss)

                sum_empirical_loss += task_empirical_loss
                sum_intra_task_comp += task_complexity

            # end inner-loop
            avg_empirical_loss = (1 / n_inner_batch) * sum_empirical_loss
            avg_intra_task_comp = (1 / n_inner_batch) * sum_intra_task_comp

            # Hyper-prior term:
            hyperprior = net_L1_norm(prior_model) * np.sqrt(1 / n_tasks) * prm.hyper_prior_factor

            # Approximated total objective:
            total_objective = avg_empirical_loss + avg_intra_task_comp + hyperprior

            # ****************************************************************************
            grad_step(total_objective, all_optimizer, lr_schedule, prm.lr, i_epoch)
            # ****************************************************************************
            # if (avg_empirical_loss.data[0] < prm.complexity_train_loss_thresh):
            #     # Take gradient step with the shared prior and all tasks' posteriors:
            #     grad_step(total_objective, all_optimizer, lr_schedule, prm.lr, i_epoch)
            # else:
            #     # Take gradient step with only tasks' posteriors to minimize the empirical loss:
            #     grad_step(avg_empirical_loss, posteriors_optimizer, lr_schedule, prm.lr, i_epoch)
            # ****************************************************************************

            # Print status:
            log_interval = 200
            if i_meta_batch % log_interval == 0:
                # TODO: average all batches and print at end of epoch... in addition to prints every number of sample batches
                batch_acc = correct_rate(outputs, targets)
                print(cmn.status_string(i_epoch, i_meta_batch, n_meta_batches, prm, batch_acc, total_objective.data[0]) +
                      'Eps-STD: {:.4}\t Empiric-Loss: {:.4}\t Intra-Comp. {:.4}\t Hyperprior: {:.4}'.
                      format(eps_std, avg_empirical_loss.data[0], avg_intra_task_comp.data[0], hyperprior.data[0]))
        # end  meta-batches loop

    # end run_epoch()

    # -------------------------------------------------------------------------------------------
    #  Test evaluation function -
    # Evaluate the mean loss on samples from the test sets of the training tasks
    # --------------------------------------------------------------------------------------------
    def run_test():
        test_acc_avg = 0.0
        n_tests = 0
        for i_task in range(n_tasks):
            model = posteriors_models[i_task]
            test_loader = train_tasks_data[i_task]['test']
            if len(test_loader) > 0:
                test_acc, test_loss = run_test_Bayes(model, test_loader, loss_criterion, prm)
                n_tests += 1
                test_acc_avg += test_acc

                n_test_samples = len(test_loader.dataset)

                print('Train Task {}, Test set: {} -  Average loss: {:.4}, Accuracy: {:.3} of {} samples\n'.format(
                    prm.test_type, i_task, test_loss, test_acc, n_test_samples))
            else:
                print('Train Task {}, Test set: {} - No test data'.
                      format(prm.test_type, i_task))

        if n_tests > 0:
            test_acc_avg /= n_tests
        return test_acc_avg

    # -----------------------------------------------------------------------------------------------------------#
    # Main script
    # -----------------------------------------------------------------------------------------------------------#

    # Update Log file
    run_name = cmn.gen_run_name('Meta-Training')
    write_result('-'*10+run_name+'-'*10, prm.log_file)
    write_result(str(prm), prm.log_file)
    write_result(cmn.get_model_string(prior_model), prm.log_file)

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
    test_acc_avg = run_test()

    # Update Log file:
    cmn.write_final_result(test_acc_avg, stop_time - start_time, prm.log_file, result_name=prm.test_type)

    # Return learned prior:
    return prior_model
