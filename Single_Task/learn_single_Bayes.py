

from __future__ import absolute_import, division, print_function

import timeit

from Models.models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import get_eps_std, run_test_Bayes
from Utils.common import grad_step, correct_rate, get_loss_criterion


def run_learning(data_loader, prm, verbose=1):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------

    # Unpack parameters:
    optim_func, optim_args, lr_schedule = \
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

    train_loader = data_loader['train']
    test_loader = data_loader['test']
    n_batches = len(train_loader)

    # get model:
    model = get_model(prm, 'Stochastic')

    #  Get optimizer:
    optimizer = optim_func(model.parameters(), **optim_args)

    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------

    def run_train_epoch(i_epoch):
        log_interval = 500

        model.train()

        for batch_idx, batch_data in enumerate(train_loader):

            eps_std = get_eps_std(i_epoch, batch_idx, n_batches, prm)

            # Monte-Carlo iterations:
            empirical_loss = 0
            n_MC = prm.n_MC if eps_std > 0 else 1
            for i_MC in range(n_MC):
                # get batch:
                inputs, targets = data_gen.get_batch_vars(batch_data, prm)

                # calculate objective:
                outputs = model(inputs, eps_std)
                empirical_loss_c = loss_criterion(outputs, targets)
                empirical_loss += (1 / n_MC) * empirical_loss_c

            # Total objective:
            objective = empirical_loss # TODO: add prior term

            # Take gradient step:
            grad_step(objective, optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            if batch_idx % log_interval == 0:
                batch_acc = correct_rate(outputs, targets)
                print(cmn.status_string(i_epoch, batch_idx, n_batches, prm, batch_acc, objective.data[0]) +
                      '\t eps-std: {:.4}'.format(eps_std))


    # -------------------------------------------------------------------------------------------
    #  Main Script
    # -------------------------------------------------------------------------------------------


    #  Update Log file
    run_name = cmn.gen_run_name('Bayes')
    if verbose == 1:
        cmn.write_result('-'*10+run_name+'-'*10, prm.log_file)
        cmn.write_result(str(prm), prm.log_file)
        cmn.write_result(cmn.get_model_string(model), prm.log_file)
        cmn.write_result('Total number of steps: {}'.format(n_batches * prm.num_epochs), prm.log_file)

    start_time = timeit.default_timer()

    # Run training epochs:
    for i_epoch in range(prm.num_epochs):
        run_train_epoch(i_epoch)

    # Test:
    test_acc, test_loss = run_test_Bayes(model, test_loader, loss_criterion, prm)

    stop_time = timeit.default_timer()
    cmn.write_final_result(test_acc, stop_time - start_time, prm.log_file, result_name=prm.test_type)

    test_err = 1 - test_acc
    return test_err