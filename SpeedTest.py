
from __future__ import absolute_import, division, print_function
import torch

import torch.optim as optim
import timeit
import argparse

import data_gen
import common as cmn
import models_Bayes
torch.backends.cudnn.benchmark=True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7

def count_correct(outputs, targets):
    pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max output
    return pred.eq(targets.data.view_as(pred)).cpu().sum()

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()


parser.add_argument('--data-source', type=str, help="Data: 'MNIST'",
                    default='MNIST')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='L2_SVM')

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--test-batch-size', type=int, help='input batch size for testing',
                    default=1000)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=200)

parser.add_argument('--lr', type=float, help='learning rate',
                    default=1e-4)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

prm = parser.parse_args()
prm.cuda = True

torch.manual_seed(prm.seed)


# Get Data:
train_loader, test_loader = data_gen.init_data_gen(prm)
n_batches = len(train_loader)

#  Get model:
model_type = 'BayesNN' # 'BayesNN' \ 'BigBayesNN'
prm.model_type = model_type
model = models_Bayes.get_model(model_type, prm)

#  Get optimizer:
optimizer = optim.Adam(model.parameters(), lr=prm.lr)

# Loss criterion
criterion = cmn.get_loss_criterion(prm.loss_type)


stage_1_ratio = 0.05  # 0.05
ratio_with_full_eps_ratio = 0.2
total_iter = prm.num_epochs * n_batches
n_iter_stage_1 = int( total_iter * stage_1_ratio)
n_iter_stage_2 = total_iter- n_iter_stage_1
n_iter_with_full_eps_std = int(n_iter_stage_2 * ratio_with_full_eps_ratio)
full_eps_std = 1.0

# -------------------------------------------------------------------------------------------
#  Training epoch  function
# -------------------------------------------------------------------------------------------

def get_eps_std(i_epoch, batch_idx):
    # We gradually increase epsilon's STD from 0 to 1.
    # The reason is that using 1 from the start results in high variance gradients.
    iter_idx = i_epoch * n_batches + batch_idx
    if iter_idx >= n_iter_stage_1:
        eps_std = full_eps_std * (iter_idx - n_iter_stage_1) / (n_iter_stage_2 - n_iter_with_full_eps_std)
    else:
        eps_std = 0.0
    eps_std = min(max(eps_std, 0.0), 1.0)  # keep in [0,1]
    return eps_std


def run_train_epoch(i_epoch):
    log_interval = 500

    model.train()

    for batch_idx, batch_data in enumerate(train_loader):

        eps_std = get_eps_std(i_epoch, batch_idx)

        # get batch:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm)

        # Take gradient step:
        optimizer.zero_grad()
        outputs = model(inputs, eps_std)
        empirical_loss = criterion(outputs, targets)

        objective = empirical_loss # TODO: add prior term

        objective.backward()
        optimizer.step()

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
    correct = 0
    for batch_data in test_loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm)
        eps_std = 0.0 # test with max-posterior
        outputs = model(inputs, eps_std)
        test_loss += criterion(outputs, targets)  # sum the mean loss in batch
        correct += count_correct(outputs, targets)

    n_test_samples = len(test_loader.dataset)
    n_test_batches = len(test_loader)
    test_loss /= n_test_batches
    test_acc = correct / n_test_samples
    print('\nTest set: Average loss: {:.4}, Accuracy: {:.3} ( {}/{})\n'.format(
        test_loss.data[0], test_acc, correct, n_test_samples))
    return test_acc


# -----------------------------------------------------------------------------------------------------------#
# Result saving
# -----------------------------------------------------------------------------------------------------------#
setting_name = 'Single_Task'
run_name = cmn.gen_run_name('Bayes')
cmn.write_result('-'*10+run_name+'-'*10, setting_name)
cmn.write_result(str(prm), setting_name)
cmn.write_result(cmn.get_model_string(model), setting_name)
cmn.write_result('Total number of steps: {}'.format(total_iter), setting_name)

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