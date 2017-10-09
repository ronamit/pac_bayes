
from __future__ import absolute_import, division, print_function

import timeit

from Models.models_standard import get_model
from Utils import common as cmn, data_gen
from Utils.common import count_correct, grad_step, correct_rate, get_loss_criterion, write_result
from Deterministic_Meta_Learning.meta_deterministic_utils import get_weights_complexity_term


def run_learning(task_data, prior_dict, prm, model_type, init_from_prior,  verbose=1):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------

    # Unpack parameters:
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

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
            intra_task_comp = get_weights_complexity_term(prm.complexity_type, prior_means_model,
                                                          prior_log_vars_model, post_model, n_train_samples)
            total_objective = empirical_loss + intra_task_comp

            # Take gradient step:
            grad_step(total_objective, optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            if batch_idx % log_interval == 0:
                batch_acc = correct_rate(outputs, targets)
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
    if verbose == 1:
        write_result('-'*10+run_name+'-'*10, prm.log_file)
        write_result(str(prm), prm.log_file)
        write_result(cmn.get_model_string(post_model), prm.log_file)        

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    for i_epoch in range(prm.num_epochs):
        run_train_epoch(i_epoch)

    # Test:
    test_acc = run_test()

    stop_time = timeit.default_timer()
    cmn.write_final_result(test_acc, stop_time - start_time, prm.log_file, result_name='Meta_Testing', verbose=verbose)

    test_err = 1 - test_acc
    return test_err