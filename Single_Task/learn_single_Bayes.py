

from __future__ import absolute_import, division, print_function

import timeit
from copy import deepcopy
from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import run_test_Bayes, get_task_complexity
from Utils.common import grad_step, correct_rate, get_value, zeros_gpu, write_to_log
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

            # Monte-Carlo iterations:
            avg_empiric_loss = zeros_gpu(1)
            n_MC = prm.n_MC

            for i_MC in range(n_MC):

                # get batch:
                inputs, targets = data_gen.get_batch_vars(batch_data, prm)
                # note: we sample new batch in eab MC run to get lower variance estimator
                batch_size = inputs.shape[0]

                # calculate objective:
                outputs = post_model(inputs)
                avg_empiric_loss_curr = (1 / batch_size) * loss_criterion(outputs, targets)
                avg_empiric_loss += (1 / n_MC) * avg_empiric_loss_curr

            # complexity/prior term:
            if prior_model:
                complexity_term = get_task_complexity(
                    prm, prior_model, post_model, n_train_samples, avg_empiric_loss)
            else:
                complexity_term = 0.0

            # Total objective:
            objective = avg_empiric_loss + complexity_term

            # Take gradient step:
            grad_step(objective, optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            log_interval = 500
            if batch_idx % log_interval == 0:
                batch_acc = correct_rate(outputs, targets)
                print(cmn.status_string(i_epoch, prm.num_epochs, batch_idx, n_batches, batch_acc, get_value(objective)) +
                      ' Loss: {:.4}\t Comp.: {:.4}'.format(get_value(avg_empiric_loss), get_value(complexity_term)))

        # End batch loop

        return
    # End run_train_epoch()
    # -------------------------------------------------------------------------------------------
    #  Main Script
    # -------------------------------------------------------------------------------------------


    #  Update Log file
    update_file = not verbose == 0
    write_to_log(cmn.get_model_string(post_model), prm, update_file=update_file)
    write_to_log('Total number of steps: {}'.format(n_batches * prm.num_epochs), prm, update_file=update_file)
    write_to_log('Number of training samples: {}'.format(data_loader['n_train_samples']), prm, update_file=update_file)

    start_time = timeit.default_timer()

    # Run training epochs:
    for i_epoch in range(prm.num_epochs):
         run_train_epoch(i_epoch)

    # evaluate final perfomance on train-set
    train_acc, train_loss = run_test_Bayes(post_model, train_loader, prm)

    # Test:
    test_acc, test_loss = run_test_Bayes(post_model, test_loader, prm)
    test_err = 1 - test_acc

    # Log resuls
    write_to_log('>Train-err. : {:.4}%\t Train-loss: {:.4} ({})'.format(100*(1-train_acc), train_loss, prm.loss_type),
                 prm, update_file=update_file)
    write_to_log('>Test-err. {:1.3}%, Test-loss:  {:.4} ({})'.format(100*(test_err), test_loss, prm.loss_type),
                 prm, update_file=update_file)

    stop_time = timeit.default_timer()
    cmn.write_final_result(test_acc, stop_time - start_time, prm)


    
 
    return post_model, test_err, test_loss


# -------------------------------------------------------------------------------------------
#  Bound evaluation
# -------------------------------------------------------------------------------------------
def eval_bound(post_model, prior_model, data_loader, prm):

    # Loss criterion
    loss_criterion = get_loss_func(prm.loss_type)

    train_loader = data_loader['train']
    n_batches = len(train_loader)
    n_train_samples = data_loader['n_train_samples']

    post_model.eval()

    empiric_loss = 0.0

    for batch_idx, batch_data in enumerate(train_loader):

        # Monte-Carlo iterations:

        n_MC = prm.n_MC

        for i_MC in range(n_MC):
            # get batch:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)
            # note: we sample new batch in each MC run to get lower variance estimator
            batch_size = inputs.shape[0]

            # calculate objective:
            outputs = post_model(inputs)
            empiric_loss += (1 / n_MC) * get_value(loss_criterion(outputs, targets))

    # End batch loop

    avg_empiric_loss = empiric_loss / n_train_samples

    #  complexity/prior term:
    complexity_term = get_task_complexity(
        prm, prior_model, post_model, n_train_samples, avg_empiric_loss)

    # Total objective:
    bound_val = avg_empiric_loss + get_value(complexity_term)
    return bound_val



# -------------------------------------------------------------------------------------------
#  Bound evaluation
# -------------------------------------------------------------------------------------------
def eval_bound2(post_model, prior_model, data_loader, prm):

    # Loss criterion
    loss_criterion = get_loss_func(prm.loss_type)

    train_loader = data_loader['train']
    n_batches = len(train_loader)
    n_train_samples = data_loader['n_train_samples']

    post_model.eval()

    avg_bound_val = 0

    for batch_idx, batch_data in enumerate(train_loader):

        # Monte-Carlo iterations:
        avg_empiric_loss = zeros_gpu(1)
        n_MC = prm.n_MC

        for i_MC in range(n_MC):
            # get batch:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)
            # note: we sample new batch in each MC run to get lower variance estimator
            batch_size = inputs.shape[0]

            # calculate objective:
            outputs = post_model(inputs)
            avg_empiric_loss_curr = (1 / batch_size) * loss_criterion(outputs, targets)
            avg_empiric_loss += (1 / n_MC) * avg_empiric_loss_curr

        #  complexity/prior term:
        complexity_term = get_task_complexity(
            prm, prior_model, post_model, n_train_samples, avg_empiric_loss)
        # TODO: maybe compute complexity  after batch loop with the total average empric error

        # Total objective:
        objective = avg_empiric_loss + complexity_term

        avg_bound_val += get_value(objective)  # save for analysis
    # End batch loop

    avg_bound_val /= n_batches

    return avg_bound_val