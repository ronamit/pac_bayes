

from __future__ import absolute_import, division, print_function
from six.moves import xrange
import timeit

import common as cmn
import data_gen
from common import count_correct, grad_step
from models_Bayes import get_bayes_model
from meta_utils import get_eps_std

def run_learning(data_loader, prm, model_type, optim_func, optim_args, loss_criterion, lr_schedule):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    train_loader = data_loader['train']
    test_loader = data_loader['test']
    n_batches = len(train_loader)

    # get model:
    model = get_bayes_model(model_type, prm)

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
                batch_acc = count_correct(outputs, targets) / prm.batch_size
                print(cmn.status_string(i_epoch, batch_idx, n_batches, prm, batch_acc, objective.data[0]) +
                      '\t eps-std: {:.4}'.format(eps_std))


    # -------------------------------------------------------------------------------------------
    #  Test evaluation function
    # -------------------------------------------------------------------------------------------
    def run_test():
        model.eval()
        test_loss = 0
        n_correct = 0
        for batch_data in test_loader:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm, is_test=True)
            eps_std = 0.0 # test with max-posterior
            outputs = model(inputs, eps_std)
            test_loss += loss_criterion(outputs, targets)  # sum the mean loss in batch
            n_correct += count_correct(outputs, targets)

        n_test_samples = len(test_loader.dataset)
        n_test_batches = len(test_loader)
        test_loss /= n_test_batches
        test_acc = n_correct / n_test_samples
        print('\nTest set: Average loss: {:.4}, Accuracy: {:.3} ( {}/{})\n'.format(
            test_loss.data[0], test_acc, n_correct, n_test_samples))
        return test_acc


    # -----------------------------------------------------------------------------------------------------------#
    #  Update Log file
    # -----------------------------------------------------------------------------------------------------------#

    run_name = cmn.gen_run_name('Bayes')
    cmn.write_result('-'*10+run_name+'-'*10, prm.log_file)
    cmn.write_result(str(prm), prm.log_file)
    cmn.write_result(cmn.get_model_string(model), prm.log_file)
    cmn.write_result(str(optim_func) + str(optim_args) +  str(lr_schedule), prm.log_file)
    cmn.write_result('Total number of steps: {}'.format(n_batches * prm.num_epochs), prm.log_file)

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
    cmn.write_final_result(test_acc, stop_time - start_time, prm.log_file)
    cmn.save_code('CodeBackup', run_name)