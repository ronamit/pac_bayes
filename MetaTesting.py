
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


def run_learning(task_data, prior_dict, prm, model_type, optim_func, optim_args, loss_criterion, lr_schedule, complexity_type, init_from_prior):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------

    # The pre-learned prior parameters are contained in these models:
    prior_means_model = prior_dict['means_model']
    prior_log_vars_model = prior_dict['log_var_model']

    # Create posterior model for the new task:
    post_model = get_model(model_type, prm)

    if init_from_prior:
        post_model.load_state_dict(prior_means_model.state_dict())

    # The data-sets of the new task:
    train_loader = task_data['train']
    test_loader = task_data['test']
    n_train_samples = len(train_loader.dataset)

    #  Get optimizer:
    optimizer = optim_func(post_model.parameters(), **optim_args)


    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------

    def run_train_epoch(i_epoch):
        log_interval = 500
        n_batches = len(train_loader)

        post_model.train()
        for batch_idx, batch_data in enumerate(train_loader):

            # get batch:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)

            # Calculate empirical loss:
            outputs = post_model(inputs)
            empirical_loss = loss_criterion(outputs, targets)

            # Total objective:
            intra_task_comp = get_intra_task_complexity(complexity_type, prior_means_model,
                                                        prior_log_vars_model, post_model, n_train_samples)
            total_objective = empirical_loss + intra_task_comp

            # Take gradient step:
            grad_step(total_objective, optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            if batch_idx % log_interval == 0:
                batch_acc = count_correct(outputs, targets) / prm.batch_size
                print(cmn.status_string(i_epoch, batch_idx, n_batches, prm, batch_acc, total_objective.data[0]))

    # -------------------------------------------------------------------------------------------
    #  Test evaluation function
    # --------------------------------------------------------------------------------------------
    def run_test():
        post_model.eval()
        test_loss = 0
        n_correct = 0
        for batch_data in test_loader:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)
            outputs = post_model(inputs)
            test_loss += loss_criterion(outputs, targets)  # sum the mean loss in batch
            n_correct += count_correct(outputs, targets)

        n_test_samples = len(test_loader.dataset)
        n_test_batches = len(test_loader)
        test_loss = test_loss.data[0] / n_test_batches
        test_acc = n_correct / n_test_samples
        print('\nTest set: Average loss: {:.4}, Accuracy: {:.3} ( {}/{})\n'.format(
            test_loss, test_acc, n_correct, n_test_samples))
        return test_acc

    # -----------------------------------------------------------------------------------------------------------#
    # Update Log file
    # -----------------------------------------------------------------------------------------------------------#
    run_name = cmn.gen_run_name('Meta-Testing')
    cmn.write_result('-'*10+run_name+'-'*10, prm.log_file)
    cmn.write_result(str(prm), prm.log_file)
    cmn.write_result(cmn.get_model_string(post_model), prm.log_file)
    cmn.write_result(str(optim_func) + str(optim_args) + str(lr_schedule), prm.log_file)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    startRuntime = timeit.default_timer()

    # Training loop:
    for i_epoch in range(prm.num_epochs):
        run_train_epoch(i_epoch)

    # Test:
    test_acc = run_test()

    stopRuntime = timeit.default_timer()
    cmn.write_final_result(test_acc, stopRuntime - startRuntime, prm.log_file)
    cmn.save_code('CodeBackup', run_name)

    return (1 - test_acc)