
from __future__ import absolute_import, division, print_function
import torch

import torch.optim as optim

import timeit
import argparse

import data_gen
import models_standard
import common as cmn
from common import count_correct, adjust_learning_rate_schedule

def run_learning(prm, model, optimizer, loss_criterion, lr_schedule):

    # Get Data:
    train_loader, test_loader = data_gen.init_data_gen(prm)

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

            # Take gradient step:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if lr_schedule:
                adjust_learning_rate_schedule(optimizer, i_epoch, prm, **lr_schedule)

            # Print status:
            if batch_idx % log_interval == 0:
                batch_acc = cmn.count_correct(outputs, targets) / prm.batch_size
                print(cmn.status_string(i_epoch, batch_idx, n_batches, prm, batch_acc, loss.data[0]))


    # -------------------------------------------------------------------------------------------
    #  Test evaluation function
    # --------------------------------------------------------------------------------------------
    def run_test():
        model.eval()
        test_loss = 0
        correct = 0
        for batch_data in test_loader:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)
            outputs = model(inputs)
            test_loss += loss_criterion(outputs, targets)  # sum the mean loss in batch
            correct += cmn.count_correct(outputs, targets)

        n_test_samples = len(test_loader.dataset)
        n_test_batches = len(test_loader)
        test_loss = test_loss.data[0] / n_test_batches
        test_acc = correct / n_test_samples
        print('\nTest set: Average loss: {:.4}, Accuracy: {:.3} ( {}/{})\n'.format(
            test_loss, test_acc, correct, n_test_samples))
        return test_acc

    # -----------------------------------------------------------------------------------------------------------#
    # Update Log file
    # -----------------------------------------------------------------------------------------------------------#
    setting_name = 'Single_Task'
    run_name = cmn.gen_run_name('Standard')
    cmn.write_result('-'*10+run_name+'-'*10, setting_name)
    cmn.write_result(str(prm), setting_name)
    cmn.write_result(cmn.get_model_string(model), setting_name)
    cmn.write_result(str(optimizer.__class__) + str(lr_schedule), setting_name)

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
    cmn.write_final_result(test_acc, stopRuntime - startRuntime, setting_name)
    cmn.save_code(setting_name, run_name)
