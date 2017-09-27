


from __future__ import absolute_import, division, print_function


from datetime import datetime
import glob
import os
import shutil

import torch.nn as nn



# -----------------------------------------------------------------------------------------------------------#
# Data manipulation
# -----------------------------------------------------------------------------------------------------------#

# torch.nn.PixelShuffle(upscale_factor)

# -----------------------------------------------------------------------------------------------------------#
# Optimizer
# -----------------------------------------------------------------------------------------------------------#

def adjust_learning_rate_interval(optimizer, epoch, prm, gamma, decay_interval):
    """Sets the learning rate to the initial LR decayed by gamma every decay_interval epochs"""
    lr = prm.lr * (gamma ** (epoch // decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_schedule(optimizer, epoch, prm, decay_factor, decay_epochs):
    """The learning rate is decayed by decay_factor at each interval start """

    # Find the index of the current interval:
    interval_index = len([mark for mark in decay_epochs if mark < epoch])

    lr = prm.lr * (decay_factor ** interval_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# -----------------------------------------------------------------------------------------------------------#
#  Configuration
# -----------------------------------------------------------------------------------------------------------#

def get_loss_criterion(loss_type):
# Note: the loss use the un-normalized net outputs (scores, not probabilities)

    criterion_dict = {'CrossEntropy':nn.CrossEntropyLoss(size_average=True),
                 'L2_SVM':nn.MultiMarginLoss(p=2, margin=1, weight=None, size_average=True)}

    return criterion_dict[loss_type]


# -----------------------------------------------------------------------------------------------------------#
# Evaluation
# -----------------------------------------------------------------------------------------------------------#
def count_correct(outputs, targets):
    pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max output
    return pred.eq(targets.data.view_as(pred)).cpu().sum()


# -----------------------------------------------------------------------------------------------------------#
# Prints
# -----------------------------------------------------------------------------------------------------------#

def status_string(i_epoch, batch_idx, n_batches, prm, batch_acc, loss_data):

    progress_per = 100. * (i_epoch * n_batches + batch_idx) / (n_batches * prm.num_epochs)
    return ('({:2.1f}%) \t Train Epoch: {:3} \t Batch: {:4} \t Loss: {:.4} \t  Acc: {:1.3}\t'.format(
        progress_per, i_epoch + 1, batch_idx, loss_data, batch_acc))

def get_model_string(model):
    return str(model.__class__)+ '\n ' + '-> '.join([m.__str__() for m in model._modules.values()])

# -----------------------------------------------------------------------------------------------------------#
# Result saving
# -----------------------------------------------------------------------------------------------------------#

def write_result(str, setting_name):

    print(str)
    with open(setting_name + '.out', 'a') as f:
        print(str, file=f)


def gen_run_name(name_prefix):
    time_str = datetime.now().strftime(' %Y-%m-%d %H:%M:%S')
    return name_prefix + time_str

def save_code(setting_name, run_name):
    dir_name = setting_name + '_' + run_name
    # Create backup of code
    source_dir = os.getcwd()
    dest_dir = source_dir + '/Code_Archive/' + dir_name
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for filename in glob.glob(os.path.join(source_dir, '*.*')):
        shutil.copy(filename, dest_dir)


def write_final_result(test_acc,run_time, setting_name):
    write_result('-'*5 + datetime.now().strftime(' %Y-%m-%d %H:%M:%S'), setting_name)
    write_result('Test Error: {:.3}%\t Runtime: {} [sec]'
                     .format(100 * (1 - test_acc), run_time, setting_name), setting_name)


