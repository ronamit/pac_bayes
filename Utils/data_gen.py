
from __future__ import absolute_import, division, print_function

import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils
from torch.autograd import Variable
import multiprocessing, os
import numpy as np
from Utils.omniglot import get_omniglot_task

# -------------------------------------------------------------------------------------------
#  Create data loader
# -------------------------------------------------------------------------------------------


def get_data_loader(prm, limit_train_samples=None, meta_split='meta_train'):

    # Set data transformation function:
    final_input_trans = None
    target_trans = []

    if prm.data_transform == 'Permute_Pixels':
        # Create a fixed random pixels permutation, applied to all images
        extra_input_trans = [create_pixel_permute_trans(prm)]

    elif prm.data_transform == 'Permute_Labels':
        # Create a fixed random label permutation, applied to all images
        target_trans = [create_label_permute_trans(prm)]

    # Get dataset:
    if prm.data_source == 'MNIST':
        train_dataset, test_dataset = load_MNIST(final_input_trans, target_trans, prm)

    elif prm.data_source == 'CIFAR10':
        train_dataset, test_dataset = load_CIFAR(final_input_trans, target_trans, prm)

    elif prm.data_source == 'Sinusoid':

        task_param = create_sinusoid_task()
        train_dataset = create_sinusoid_data(task_param, n_samples=10)
        test_dataset = create_sinusoid_data(task_param, n_samples=100)

    elif prm.data_source == 'Omniglot':
        train_dataset, test_dataset = get_omniglot_task(prm.data_path, meta_split,
            n_labels=prm.n_way_k_shot['N'], k_train_shot=prm.n_way_k_shot['K'],
            final_input_trans=final_input_trans, target_transform=target_trans)
    else:
        raise ValueError('Invalid data_source')

    # Limit the training samples:
    n_train_samples_orig = len(train_dataset)
    if limit_train_samples and limit_train_samples < n_train_samples_orig:
        sampled_inds = torch.randperm(n_train_samples_orig)[:limit_train_samples]
        train_dataset.train_data = train_dataset.train_data[sampled_inds]
        train_dataset.train_labels = train_dataset.train_labels[sampled_inds]


    # Create data loaders:
    kwargs = {'num_workers': multiprocessing.cpu_count(), 'pin_memory': True} if prm.cuda else {}

    train_loader = data_utils.DataLoader(train_dataset, batch_size=prm.batch_size, shuffle=True, **kwargs)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=prm.test_batch_size, shuffle=True, **kwargs)

    n_train_samples = len(train_loader.dataset)
    n_test_samples = len(test_loader.dataset)

    data_loader = {'train': train_loader, 'test': test_loader,
                   'n_train_samples': n_train_samples, 'n_test_samples': n_test_samples}

    return data_loader


# -------------------------------------------------------------------------------------------
#  MNIST  Data set
# -------------------------------------------------------------------------------------------

def load_MNIST(final_input_trans, target_trans, prm):

    # Data transformations list:
    transform = [transforms.ToTensor()]

    # Normalize values:
    # Note: original values  in the range [0,1]

    # MNIST_MEAN = (0.1307,)  # (0.5,)
    # MNIST_STD = (0.3081,)  # (0.5,)
    # transform += transforms.Normalize(MNIST_MEAN, MNIST_STD)

    transform += [transforms.Normalize((0.5,), (0.5,))]  # transform to [-1,1]

    if final_input_trans:
        transform += final_input_trans

    root_path = os.path.join(prm.data_path, 'MNIST')

    # Train set:
    train_dataset = datasets.MNIST(root_path, train=True, download=True,
                                   transform=transforms.Compose(transform), target_transform=transforms.Compose(target_trans))

    # Test set:
    test_dataset = datasets.MNIST(root_path, train=False,
                                  transform=transforms.Compose(transform), target_transform=transforms.Compose(target_trans))


    return train_dataset, test_dataset



def load_CIFAR(final_input_trans, target_trans, prm):

    # Data transformations list:
    transform = [transforms.ToTensor()]

    # Normalize values:
    # Note: original values  in the range [0,1]
    transform += [transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]  # transform to [-1,1]

    if final_input_trans:
        transform += final_input_trans

    root_path = os.path.join(prm.data_path, 'CIFAR10')

    # Train set:
    train_dataset = datasets.CIFAR10(root_path, train=True, download=True,
                                   transform=transforms.Compose(transform), target_transform=transforms.Compose(target_trans))

    # Test set:
    test_dataset = datasets.CIFAR10(root_path, train=False,
                                  transform=transforms.Compose(transform), target_transform=transforms.Compose(target_trans))


    return train_dataset, test_dataset

# -------------------------------------------------------------------------------------------
#  Data sets parameters
# -------------------------------------------------------------------------------------------


def get_info(prm):
    if prm.data_source == 'MNIST':
        info = {'input_shape': (1, 28, 28),  'n_classes': 10}
    elif prm.data_source == 'CIFAR10':
        info = {'input_shape': (3, 32, 32), 'n_classes': 10}
    elif prm.data_source == 'Omniglot':
        info = {'input_shape': (1, 28, 28), 'n_classes': prm.n_way_k_shot['N']}

    else:
        raise ValueError('Invalid data_source')

    return info


# -------------------------------------------------------------------------------------------
#  Transform batch to variables
# -------------------------------------------------------------------------------------------
def get_batch_vars(batch_data, args, is_test=False):
    inputs, targets = batch_data
    if args.cuda:
        inputs, targets = inputs.cuda(), targets.cuda(async=True)
    inputs, targets = Variable(inputs, volatile=is_test), Variable(targets, volatile=is_test)
    return inputs, targets

# -----------------------------------------------------------------------------------------------------------#
# Data manipulation
# -----------------------------------------------------------------------------------------------------------#

def create_pixel_permute_trans(prm):
    info = get_info(prm)
    input_shape = info['input_shape']
    input_size = input_shape[0] * input_shape[1] * input_shape[2]
    inds_permute = torch.randperm(input_size)
    transform_func = lambda x: permute_pixels(x, inds_permute)
    return transform_func

def permute_pixels(x, inds_permute):
    ''' Permute pixels of a tensor image'''
    im_H = x.shape[1]
    im_W = x.shape[2]
    input_size = im_H * im_W
    x = x.view(input_size)  # flatten image
    x = x[inds_permute]
    x = x.view(1, im_H, im_W)
    return x

def create_label_permute_trans(prm):
    info = get_info(prm)
    inds_permute = torch.randperm(info['n_classes'])
    transform_func = lambda target: inds_permute[target]
    return transform_func

# -----------------------------------------------------------------------------------------------------------#
# Sinusoid Regression
# -----------------------------------------------------------------------------------------------------------#
def create_sinusoid_task():
    task_param = {'phase':np.random.uniform(0, np.pi),
                  'amplitude':np.random.uniform(0.1, 5.0),
                  'freq': 5.0,
                  'input_range': [-0.5, 0.5]}
    return task_param

def create_sinusoid_data(task_param, n_samples):
    amplitude = task_param['amplitude']
    phase = task_param['phase']
    freq = task_param['freq']
    input_range = task_param['input_range']
    y = np.ndarray(shape=(n_samples, 1), dtype=np.float32)
    x = np.random.uniform(input_range[0], input_range[1], n_samples)
    y = amplitude * np.sin(phase + 2 * np.pi * freq * x)
    return x, y