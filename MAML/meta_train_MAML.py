
#
# the code is inspired by: https://github.com/katerakelly/pytorch-maml


from __future__ import absolute_import, division, print_function

import timeit
import random
import numpy as np
from collections import OrderedDict

import torch
from torch.autograd import Variable
from Models.func_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import get_posterior_complexity_term, run_test_Bayes
from Utils.common import grad_step, net_norm, correct_rate, get_loss_criterion, write_result, count_correct


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

    # Create a 'dummy' model to generate the set of parameters of the shared initial point (theta):
    model = get_model(prm)
    model.train()

    # Create optimizer for meta-params (theta)
    meta_params = list(model.parameters())

    meta_optimizer = optim_func(meta_params, **optim_args)

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

            meta_batch_start = meta_batch_starts[i_meta_batch]
            task_ids_in_meta_batch = task_order[meta_batch_start: (meta_batch_start + prm.meta_batch_size)]
            n_inner_batch = len(task_ids_in_meta_batch)  # it may be less than  prm.meta_batch_size at the last one
            # note: it is OK if some task appear several times in the meta-batch

            total_objective = 0
            correct_count = 0
            sample_count = 0
            # samples-batches loop (inner loop) - updates weights with gradient steps on training data
            for i_inner_batch in range(n_inner_batch):

                task_id = task_ids_in_meta_batch[i_inner_batch]

                # get sample-batch data from current task to calculate the empirical loss estimate:
                try:
                    batch_data = task_train_loaders[task_id].next()
                except StopIteration:
                    # in case some task has less samples - just restart the iterator and re-use the samples
                    task_train_loaders[task_id] = iter(train_tasks_data[task_id]['train'])
                    batch_data = task_train_loaders[task_id].next()

                fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())

                for i_step in range(prm.n_meta_train_grad_steps):

                    # get batch variables:
                    inputs, targets = data_gen.get_batch_vars(batch_data, prm)

                    if i_step == 0:
                        outputs = model(inputs)
                    else:
                        outputs = model(inputs, fast_weights)
                    # Empirical Loss on current task:
                    task_loss = loss_criterion(outputs, targets)
                    grads = torch.autograd.grad(task_loss, fast_weights.values(), create_graph=True)

                    fast_weights = OrderedDict((name, param - prm.alpha * grad)
                                               for ((name, param), grad) in zip(fast_weights.items(), grads))
                #  end grad steps loop

                # Sample new  (validation) data batch for this task:
                inputs, targets = data_gen.get_batch_vars(batch_data, prm)
                outputs = model(inputs, fast_weights)
                total_objective += loss_criterion(outputs, targets)
                correct_count += count_correct(outputs, targets)
                sample_count += inputs.size(0)
            # end inner-loop

            # Take gradient step with the meta-parameters (theta) based on validation data:
            grad_step(total_objective, meta_optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            log_interval = 200
            if i_meta_batch % log_interval == 0:
                # TODO: average all batches and print at end of epoch... in addition to prints every number of sample batches
                batch_acc = correct_count / sample_count
                print(cmn.status_string(i_epoch, i_meta_batch, n_meta_batches, prm, batch_acc, total_objective.data[0]))
        # end  meta-batches loop

    # end run_epoch()



    # -----------------------------------------------------------------------------------------------------------#
    # Main script
    # -----------------------------------------------------------------------------------------------------------#

    # Update Log file
    run_name = cmn.gen_run_name('Meta-Training')
    write_result('-'*10 + run_name + '-'*10, prm.log_file)
    write_result(str(prm), prm.log_file)
    write_result(cmn.get_model_string(model), prm.log_file)

    write_result('---- Meta-Training set: {0} tasks'.format(len(train_tasks_data)), prm.log_file)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    for i_epoch in range(prm.num_epochs):
        run_train_epoch(i_epoch)

    stop_time = timeit.default_timer()

    # Update Log file:
    cmn.write_final_result(0.0, stop_time - start_time, prm.log_file)

    # Return learned meta-parameters:
    return model
