
from __future__ import absolute_import, division, print_function

import timeit
import random
import numpy as np

from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import get_posterior_complexity_term, run_test_Bayes
from Utils.common import grad_step, net_norm, count_correct, get_loss_criterion, write_result


# -------------------------------------------------------------------------------------------
# function for meta step
# -------------------------------------------------------------------------------------------
def meta_step(prior_model, prm, mb_data_loaders, mb_iterators, mb_posteriors_models, loss_criterion, n_train_tasks):
    '''  Calculate objective based on tasks in meta-batch '''
    # note: it is OK if some tasks appear several times in the meta-batch

    n_tasks_in_mb = len(mb_data_loaders)

    sum_empirical_loss = 0
    sum_intra_task_comp = 0
    correct_count = 0
    sample_count = 0

    # ----------- loop over tasks in meta-batch -----------------------------------#
    for i_task in range(n_tasks_in_mb):

        n_samples = mb_data_loaders[i_task]['n_train_samples']

        # get sample-batch data from current task to calculate the empirical loss estimate:
        batch_data = data_gen.get_next_batch_cyclic(mb_iterators[i_task], mb_data_loaders[i_task]['train'])

        # The posterior model corresponding to the task in the batch:
        post_model = mb_posteriors_models[i_task]
        post_model.train()

        # Monte-Carlo iterations:
        n_MC = prm.n_MC
        task_empirical_loss = 0
        task_complexity = 0
        # ----------- Monte-Carlo loop  -----------------------------------#
        for i_MC in range(n_MC):
            # get batch variables:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)

            # Empirical Loss on current task:
            outputs = post_model(inputs)
            curr_empirical_loss = loss_criterion(outputs, targets)
            task_empirical_loss += (1 / n_MC) * curr_empirical_loss

            correct_count += count_correct(outputs, targets)
            sample_count += inputs.size(0)

            # Intra-task complexity of current task:
            curr_complexity = get_posterior_complexity_term(
                prm, prior_model, post_model,
                n_samples, curr_empirical_loss, noised_prior=True)
            task_complexity += (1 / n_MC) * curr_complexity
        # end Monte-Carlo loop

        sum_empirical_loss += task_empirical_loss
        sum_intra_task_comp += task_complexity

    # end loop over tasks in meta-batch
    avg_empirical_loss = (1 / n_tasks_in_mb) * sum_empirical_loss
    avg_intra_task_comp = (1 / n_tasks_in_mb) * sum_intra_task_comp

    # Hyper-prior term:
    hyperprior = net_norm(prior_model, p=2) * np.sqrt(1 / n_train_tasks) * prm.hyperprior_factor

    # Approximated total objective:
    total_objective = avg_empirical_loss + avg_intra_task_comp + hyperprior

    info = {'sample_count': sample_count, 'correct_count': correct_count,
                  'avg_empirical_loss': avg_empirical_loss.data[0],
                  'avg_intra_task_comp': avg_intra_task_comp.data[0],
                  'hyperprior': hyperprior.data[0]}
    return total_objective, info

# -------------------------------------------------------------------------------------------
#  Learning function
# -------------------------------------------------------------------------------------------


def run_meta_learning(data_loaders, prm):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    # Unpack parameters:
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

    n_train_tasks = len(data_loaders)

    # Create posterior models for each task:
    posteriors_models = [get_model(prm) for _ in range(n_train_tasks)]

    # Create a 'dummy' model to generate the set of parameters of the shared prior:
    prior_model = get_model(prm)

    # Gather all tasks posterior params:
    all_post_param = []
    for i_task in range(n_train_tasks):
        post_params = list(posteriors_models[i_task].parameters())
        all_post_param += post_params

    # Create optimizer for all parameters (posteriors + prior)
    prior_params = list(prior_model.parameters())
    all_params = all_post_param + prior_params
    all_optimizer = optim_func(all_params, **optim_args)

    # number of sample-batches in each task:
    n_batch_list = [len(data_loader['train']) for data_loader in data_loaders]

    n_batches_per_task = np.max(n_batch_list)

    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------
    def run_train_epoch(i_epoch):

        # For each task, prepare an iterator to generate training batches:
        train_iterators = [iter(data_loaders[ii]['train']) for ii in range(n_train_tasks)]

        # The task order to take batches from:
        # The meta-batch will be balanced - i.e, each task will appear roughly the same number of times
        # note: if some tasks have less data that other tasks - it may be sampled more than once in an epoch
        task_order = []
        task_ids_list = list(range(n_train_tasks))
        for i_batch in range(n_batches_per_task):
            random.shuffle(task_ids_list)
            task_order += task_ids_list
        # Note: this method ensures each training sample in each task is drawn in each epoch.
        # If all the tasks have the same number of sample, then each sample is drawn exactly once in an epoch.

        # ----------- meta-batches loop (batches of tasks) -----------------------------------#
        # each meta-batch includes several tasks
        # we take a grad step with theta after each meta-batch
        meta_batch_starts = list(range(0, len(task_order), prm.meta_batch_size))
        n_meta_batches = len(meta_batch_starts)

        for i_meta_batch in range(n_meta_batches):


            meta_batch_start = meta_batch_starts[i_meta_batch]
            task_ids_in_meta_batch = task_order[meta_batch_start: (meta_batch_start + prm.meta_batch_size)]
            # meta-batch size may be less than  prm.meta_batch_size at the last one
            # note: it is OK if some tasks appear several times in the meta-batch

            mb_data_loaders = [data_loaders[task_id] for task_id in task_ids_in_meta_batch]
            mb_iterators = [train_iterators[task_id] for task_id in task_ids_in_meta_batch]
            mb_posteriors_models = [posteriors_models[task_id] for task_id in task_ids_in_meta_batch]

            # Get objective based on tasks in meta-batch:
            total_objective, info = meta_step(prior_model, prm, mb_data_loaders,
                                              mb_iterators, mb_posteriors_models, loss_criterion, n_train_tasks)

            # Take gradient step with the shared prior and all tasks' posteriors:
            grad_step(total_objective, all_optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            log_interval = 200
            if i_meta_batch % log_interval == 0:
                batch_acc = info['correct_count'] / info['sample_count']
                print(cmn.status_string(i_epoch,  prm.num_epochs, i_meta_batch, n_meta_batches, batch_acc, total_objective.data[0]) +
                      ' Empiric-Loss: {:.4}\t Intra-Comp. {:.4}\t Hyperprior: {:.4}'.
                      format(info['avg_empirical_loss'], info['avg_intra_task_comp'], info['hyperprior']))
        # end  meta-batches loop

    # end run_epoch()

    # -------------------------------------------------------------------------------------------
    #  Test evaluation function -
    # Evaluate the mean loss on samples from the test sets of the training tasks
    # --------------------------------------------------------------------------------------------
    def run_test():
        test_acc_avg = 0.0
        n_tests = 0
        for i_task in range(n_train_tasks):
            model = posteriors_models[i_task]
            test_loader = data_loaders[i_task]['test']
            if len(test_loader) > 0:
                test_acc, test_loss = run_test_Bayes(model, test_loader, loss_criterion, prm)
                n_tests += 1
                test_acc_avg += test_acc

                n_test_samples = len(test_loader.dataset)

                write_result('Train Task {}, Test set: {} -  Average loss: {:.4}, Accuracy: {:.3} of {} samples\n'.format(
                    prm.test_type, i_task, test_loss, test_acc, n_test_samples), prm.log_file)
            else:
                print('Train Task {}, Test set: {} - No test data'.format(prm.test_type, i_task))

        if n_tests > 0:
            test_acc_avg /= n_tests
        return test_acc_avg

    # -----------------------------------------------------------------------------------------------------------#
    # Main script
    # -----------------------------------------------------------------------------------------------------------#

    # Update Log file
    run_name = cmn.gen_run_name('Meta-Training')
    write_result('-'*10 + run_name + '-'*10, prm.log_file)
    write_result(str(prm), prm.log_file)
    write_result(cmn.get_model_string(prior_model), prm.log_file)

    write_result('---- Meta-Training set: {0} tasks'.format(len(data_loaders)), prm.log_file)

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
