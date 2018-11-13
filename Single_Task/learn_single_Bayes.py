

from __future__ import absolute_import, division, print_function

import timeit
from copy import deepcopy
import torch
from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import run_eval_Bayes
from Utils.complexity_terms import get_task_complexity
from Utils.common import grad_step, correct_rate, write_to_log
from Utils.Losses import get_loss_func

# -------------------------------------------------------------------------------------------
#  Stochastic Single-task learning
# -------------------------------------------------------------------------------------------

def run_learning(data_loader, prm, prior_model=None, init_from_prior=True, verbose=1):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------

    # Unpack parameters:
    optim_func, optim_args, lr_schedule = \
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_func(prm.loss_type)

    train_loader = data_loader['train']
    test_loader = data_loader['test']
    n_batches = len(train_loader)
    n_train_samples = data_loader['n_train_samples']

    # get model:
    if prior_model and init_from_prior:
        # init from prior model:
        post_model = deepcopy(prior_model)
    else:
        post_model = get_model(prm)

    # post_model.set_eps_std(0.0) # DEBUG: turn off randomness

    #  Get optimizer:
    optimizer = optim_func(post_model.parameters(), **optim_args)

    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------

    def run_train_epoch(i_epoch):

        # # Adjust randomness (eps_std)
        # if hasattr(prm, 'use_randomness_schedeule') and prm.use_randomness_schedeule:
        #     if i_epoch > prm.randomness_full_epoch:
        #         eps_std = 1.0
        #     elif i_epoch > prm.randomness_init_epoch:
        #         eps_std = (i_epoch - prm.randomness_init_epoch) / (prm.randomness_full_epoch - prm.randomness_init_epoch)
        #     else:
        #         eps_std = 0.0  #  turn off randomness
        #     post_model.set_eps_std(eps_std)

        # post_model.set_eps_std(0.00) # debug

        post_model.train()

        for batch_idx, batch_data in enumerate(train_loader):

            # get batch data:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)

            batch_size = inputs.shape[0]

            # Monte-Carlo iterations:
            avg_empiric_loss = torch.zeros(1, device=prm.device)
            n_MC = prm.n_MC

            for i_MC in range(n_MC):

                # calculate objective:
                outputs = post_model(inputs)
                avg_empiric_loss_curr = (1 / batch_size) * loss_criterion(outputs, targets)
                avg_empiric_loss += (1 / n_MC) * avg_empiric_loss_curr

            # complexity/prior term:
            if prior_model:
                complexity_term = get_task_complexity(
                    prm, prior_model, post_model, n_train_samples, avg_empiric_loss)
            else:
                complexity_term = torch.zeros(1, device=prm.device)

            # Total objective:
            objective = avg_empiric_loss + complexity_term

            # Take gradient step:
            grad_step(objective, optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            log_interval = 1000
            if batch_idx % log_interval == 0:
                batch_acc = correct_rate(outputs, targets)
                print(cmn.status_string(i_epoch, prm.num_epochs, batch_idx, n_batches, batch_acc, objective.item()) +
                      ' Loss: {:.4}\t Comp.: {:.4}'.format(avg_empiric_loss.item(), complexity_term.item()))

        # End batch loop

        return
    # End run_train_epoch()
    # -------------------------------------------------------------------------------------------
    #  Main Script
    # -------------------------------------------------------------------------------------------


    #  Update Log file
    if verbose:
        write_to_log(cmn.get_model_string(post_model), prm)
        write_to_log('Number of weights: {}'.format(post_model.weights_count), prm)
        write_to_log('Total number of steps: {}'.format(n_batches * prm.num_epochs), prm)
        write_to_log('Number of training samples: {}'.format(data_loader['n_train_samples']), prm)


    start_time = timeit.default_timer()

    # Run training epochs:
    for i_epoch in range(prm.num_epochs):
         run_train_epoch(i_epoch)

    # evaluate final perfomance on train-set
    train_acc, train_loss = run_eval_Bayes(post_model, train_loader, prm)

    # Test:
    test_acc, test_loss = run_eval_Bayes(post_model, test_loader, prm)
    test_err = 1 - test_acc

    # Log results
    if verbose:
        write_to_log('>Train-err. : {:.4}%\t Train-loss: {:.4}'.format(100*(1-train_acc), train_loss), prm)
        write_to_log('>Test-err. {:1.3}%, Test-loss:  {:.4}'.format(100*(test_err), test_loss), prm)

    stop_time = timeit.default_timer()
    if verbose:
        cmn.write_final_result(test_acc, stop_time - start_time, prm)

    return post_model, test_err, test_loss


# -------------------------------------------------------------------------------------------
#  Bound evaluation
# -------------------------------------------------------------------------------------------
def eval_bound(post_model, prior_model, data_loader, prm, avg_empiric_loss=None):

    # # Loss criterion
    # loss_criterion = get_loss_func(prm.loss_type)
    #
    # train_loader = data_loader['train']
    # n_batches = len(train_loader)
    n_train_samples = data_loader['n_train_samples']
    #
    # post_model.eval()
    #
    # empiric_loss = 0.0
    #
    # for batch_idx, batch_data in enumerate(train_loader):
    #
    #     # get batch:
    #     inputs, targets = data_gen.get_batch_vars(batch_data, prm)
    #     batch_size = inputs.shape[0]
    #
    #     # Monte-Carlo iterations:
    #     n_MC = prm.n_MC_eval
    #     for i_MC in range(n_MC):
    #
    #         # calculate loss:
    #         outputs = post_model(inputs)
    #         empiric_loss += (1 / n_MC) * loss_criterion(outputs, targets).item()
    #
    # # End batch loop
    #
    # avg_empiric_loss = empiric_loss / n_train_samples

    if not avg_empiric_loss:
        _, avg_empiric_loss = run_eval_Bayes(post_model, data_loader['train'], prm)


    #  complexity/prior term:
    complexity_term = get_task_complexity(
        prm, prior_model, post_model, n_train_samples, avg_empiric_loss)

    # Total objective:
    bound_val = avg_empiric_loss + complexity_term.item()
    return bound_val


