


from __future__ import absolute_import, division, print_function


from datetime import datetime
import glob
import os
import shutil

import torch.nn as nn

# torch.nn.PixelShuffle(upscale_factor)


def status_string(i_epoch, batch_idx, n_batches, prm, batch_acc, loss_data):

    progress_per = 100. * (i_epoch * n_batches + batch_idx) / (n_batches * prm.num_epochs)
    return ('({:2.1f}%) \t Train Epoch: {:3} \t Batch: {:4} \t Loss: {:.4} \t  Acc: {:1.3}\t'.format(
        progress_per, i_epoch + 1, batch_idx, loss_data, batch_acc))


def get_loss_criterion(loss_type):
# Note: the loss use the un-normalized net outputs (scores, not probabilities)

    if loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(size_average=True)

    elif loss_type == 'L2_SVM':
        criterion = nn.MultiMarginLoss(p=2, margin=1, weight=None, size_average=True)
    else:
        raise  ValueError('Invalid loss_type')
    return criterion


def count_correct(outputs, targets):
    pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max output
    return pred.eq(targets.data.view_as(pred)).cpu().sum()

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

