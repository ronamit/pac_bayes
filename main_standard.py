
from __future__ import absolute_import, division, print_function
import torch

import torch.optim as optim

import timeit
import argparse

import data_gen
import models_standard
import common as cmn

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()


parser.add_argument('--data-source', type=str, help="Data: 'MNIST'",
                    default='MNIST')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--test-batch-size', help='input batch size for testing',
                    type=int, default=1000)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=10)

parser.add_argument('--lr', type=float, help='learning rate',
                    default=1e-4)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
prm = parser.parse_args()
prm.cuda = not prm.no_cuda and torch.cuda.is_available()

torch.manual_seed(prm.seed)

# Get Data:
train_loader, test_loader = data_gen.init_data_gen(prm)

#  Get model:
model_type = 'ConvNet' # 'FcNet' \ 'ConvNet'
model = models_standard.get_model(model_type, prm)

#  Get optimizer:
optimizer = optim.Adam(model.parameters(), lr=prm.lr)

# Loss criterion
criterion = cmn.get_loss_criterion(prm.loss_type)

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
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

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
        test_loss += criterion(outputs, targets)  # sum the mean loss in batch
        correct += cmn.count_correct(outputs, targets)

    n_test_samples = len(test_loader.dataset)
    n_test_batches = len(test_loader)
    test_loss = test_loss.data[0] / n_test_batches
    test_acc = correct / n_test_samples
    print('\nTest set: Average loss: {:.4}, Accuracy: {:.3} ( {}/{})\n'.format(
        test_loss, test_acc, correct, n_test_samples))
    return test_acc

# -----------------------------------------------------------------------------------------------------------#
# Result saving
# -----------------------------------------------------------------------------------------------------------#
setting_name = 'Single_Task'
run_name = cmn.gen_run_name('Standard')
cmn.write_result('-'*10+run_name+'-'*10, setting_name)
cmn.write_result(str(prm), setting_name)
cmn.write_result(cmn.get_model_string(model), setting_name)
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

cmn.write_result('Test Error: {:.3}%\t Runtime: {:.3} [sec]'
             .format(100*(1-test_acc), stopRuntime - startRuntime), setting_name)

cmn.save_code(setting_name, run_name)
