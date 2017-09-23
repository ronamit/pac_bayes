

from __future__ import absolute_import, division, print_function
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable


test_batch_size = 1000


def init_data_gen(prm):

    kwargs = {'num_workers': 1, 'pin_memory': True} if prm.cuda else {}

    if prm.data_source == 'MNIST':
        # MNIST_MEAN =  (0.1307,) # (0.5,)
        # MNIST_STD =  (0.3081,)  # (0.5,)
        # Note: keep values in [0,1] to avoid too large input norm (which cause high variance)
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               # transforms.Normalize(MNIST_MEAN, MNIST_STD)
                           ])),
            batch_size=prm.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               # transforms.Normalize(MNIST_MEAN, MNIST_STD)
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

    else:
        raise ValueError('Invalid data_source')

    return train_loader, test_loader


def get_batch_vars(batch_data, args):
    inputs, targets = batch_data
    if args.cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    return inputs, targets


def get_info(prm):
    if prm.data_source == 'MNIST':
        info = {'im_size': 28, 'color_channels': 1, 'n_classes': 10, 'input_size': 1 * 28 * 28}
    else:
        raise ValueError('Invalid data_source')

    return info
