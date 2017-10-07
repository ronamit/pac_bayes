
from __future__ import absolute_import, division, print_function

import timeit
import data_gen

import numpy as np
import torch
import common as cmn
from common import count_correct, get_param_from_model, grad_step, correct_rate
from models_Bayes import get_bayes_model
from Bayes_utils import get_posterior_complexity_term, get_eps_std, run_test_Bayes


def run_learning(task_data, prior_model, prm, model_type, optim_func, optim_args,
                 loss_criterion, lr_schedule, init_from_prior, verbose=1):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------


    # Create posterior model for the new task:
    post_model = get_bayes_model(model_type, prm)

    if init_from_prior:
        post_model.load_state_dict(prior_model.state_dict())

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

            eps_std = get_eps_std(i_epoch, batch_idx, n_batches, prm)

            # Monte-Carlo iterations:
            n_MC = prm.n_MC if eps_std > 0 else 1
            empirical_loss = 0
            for i_MC in range(n_MC):
                # get batch:
                inputs, targets = data_gen.get_batch_vars(batch_data, prm)

                # Calculate empirical loss:
                outputs = post_model(inputs, eps_std)
                empirical_loss += (1 / n_MC) * loss_criterion(outputs, targets)

            # Total objective:
            intra_task_comp = get_posterior_complexity_term(
                prm.complexity_type, prior_model, post_model, n_train_samples)
            total_objective = empirical_loss + intra_task_comp

            # Take gradient step:
            grad_step(total_objective, optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            if batch_idx % log_interval == 0:
                batch_acc = correct_rate(outputs, targets)
                print(cmn.status_string(i_epoch, batch_idx, n_batches, prm, batch_acc, total_objective.data[0]) +
                      'Eps-STD: {:.4}\t Empiric Loss: {:.4}\t Intra-Comp. {:.4}'.
                      format(eps_std, empirical_loss.data[0], intra_task_comp.data[0]))


    # -----------------------------------------------------------------------------------------------------------#
    # Update Log file
    # -----------------------------------------------------------------------------------------------------------#
    run_name = cmn.gen_run_name('Meta-Testing')
    if verbose == 1:
        cmn.write_result('-'*10+run_name+'-'*10, prm.log_file)
        cmn.write_result(str(prm), prm.log_file)
        cmn.write_result(model_type, prm.log_file)
        cmn.write_result(str(optim_func) + str(optim_args) + str(lr_schedule), prm.log_file)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    for i_epoch in range(prm.num_epochs):
        run_train_epoch(i_epoch)

    # Test:
    test_acc, test_loss = run_test_Bayes(post_model, test_loader, loss_criterion, prm)

    stop_time = timeit.default_timer()
    cmn.write_final_result(test_acc, stop_time - start_time, prm.log_file, result_name=prm.test_type, verbose=verbose)


    return (1 - test_acc)