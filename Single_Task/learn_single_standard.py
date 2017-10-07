
from __future__ import absolute_import, division, print_function

import timeit

from Models.models_standard import get_model
from Utils import common as cmn, data_gen
from Utils.common import count_correct, grad_step, correct_rate


def run_learning(data_loader, prm, model_type, optim_func, optim_args, loss_criterion, lr_schedule, verbose=1):

    # The data-sets:
    train_loader = data_loader['train']
    test_loader = data_loader['test']

    # Create  model:
    model = get_model(model_type, prm)

    #  Get optimizer:
    optimizer = optim_func(model.parameters(), **optim_args)

    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------

    def run_train_epoch(i_epoch):
        log_interval = 500
        n_batches = len(train_loader)

        model.train()
        for batch_idx, batch_data in enumerate(train_loader):

            # get batch:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)

            # Calculate loss:
            outputs = model(inputs)
            loss = loss_criterion(outputs, targets)

            # Take gradient step:
            grad_step(loss, optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            if batch_idx % log_interval == 0:
                batch_acc = correct_rate(outputs, targets)
                print(cmn.status_string(i_epoch, batch_idx, n_batches, prm, batch_acc, loss.data[0]))


    # -------------------------------------------------------------------------------------------
    #  Test evaluation function
    # --------------------------------------------------------------------------------------------
    def run_test():
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
        print('\nTest set: Average loss: {:.4}, Accuracy: {:.3} ( {}/{})\n'.format(
            test_loss, test_acc, n_correct, n_test_samples))
        return test_acc

    # -----------------------------------------------------------------------------------------------------------#
    # Update Log file
    # -----------------------------------------------------------------------------------------------------------#
    run_name = cmn.gen_run_name('Standard')
    if verbose == 1:
        cmn.write_result('-'*10+run_name+'-'*10, prm.log_file)
        cmn.write_result(str(prm), prm.log_file)
        cmn.write_result(cmn.get_model_string(model), prm.log_file)
        cmn.write_result(str(optim_func) + str(optim_args) + str(lr_schedule), prm.log_file)

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
    cmn.write_final_result(test_acc, stop_time - start_time, prm.log_file, verbose=verbose)

    return (1-test_acc)
