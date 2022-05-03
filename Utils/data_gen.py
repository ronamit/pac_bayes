import os
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from Utils.real_datasets import fetch_MUSHROOMS, fetch_TICTACTOE, fetch_HABERMAN, fetch_PHISHING, fetch_ADULT, \
    fetch_CODRNA, fetch_SVMGUIDE1


# See https://papers.nips.cc/paper/2021/file/0415740eaa4d9decbc8da001d3fd805f-Supplemental.pdf
BINARY_DATASETS = {
    'MUSHROOMS': fetch_MUSHROOMS,
    'TICTACTOE': fetch_TICTACTOE,
    'HABERMAN': fetch_HABERMAN,
    'PHISHING': fetch_PHISHING,
    'ADULT': fetch_ADULT,
    'CODRNA': fetch_CODRNA,
    'SVMGUIDE': fetch_SVMGUIDE1
}
BINARY_DATASETS_DIM = {
    'MUSHROOMS': 22,
    'TICTACTOE': 9,
    'HABERMAN': 3,
    'PHISHING': 68,
    'ADULT': 123,
    'CODRNA': 8,
    'SVMGUIDE': 4
}
# -------------------------------------------------------------------------------------------
#  Task generator class
# -------------------------------------------------------------------------------------------

class Task_Generator(object):

    def __init__(self, prm):

        self.data_source = prm.data_source
        self.data_transform = prm.data_transform
        self.data_path = prm.data_path

    def get_data_loader(self, prm, limit_train_samples=None):

        # Set data transformation function:
        if self.data_transform == 'Permute_Pixels':
            # Create a fixed random pixels permutation, applied to all images
            final_input_trans = [create_pixel_permute_trans(prm)]
            target_trans = []
        elif self.data_transform == 'Shuffled_Pixels':
            # Create a fixed random pixels permutation, applied to all images
            final_input_trans = [create_limited_pixel_permute_trans(prm)]
            target_trans = []
        elif self.data_transform == 'Permute_Labels':
            # Create a fixed random label permutation, applied to all images
            target_trans = [create_label_permute_trans(prm)]
            final_input_trans = None
        elif self.data_transform == 'Rotate90':
            # all images in task are rotated by some random angle from [0,90,180,270]
            final_input_trans = [create_rotation_trans()]
            target_trans = []
        elif self.data_transform == 'None':
            final_input_trans = None
            target_trans = []
        else:
            raise ValueError('Unrecognized data_transform')
        # Get dataset:
        if self.data_source == 'MNIST':
            train_dataset, test_dataset = load_MNIST(final_input_trans, target_trans, prm)
        elif self.data_source == 'CIFAR10':
            train_dataset, test_dataset = load_CIFAR(final_input_trans, target_trans, prm)
        elif self.data_source == 'binarized_MNIST':
            assert not target_trans  # make sure no transformations
            target_trans = [create_label_binarize(prm, thresh=5)]
            train_dataset, test_dataset = load_MNIST(final_input_trans, target_trans, prm)
        elif self.data_source in BINARY_DATASETS:
            fetch_fn = BINARY_DATASETS[self.data_source]
            path = Path(prm.data_path) / self.data_source
            data_dict = fetch_fn(path, valid_size=0., test_size=0.2, seed=prm.seed)
            train_dataset = TensorDataset(torch.Tensor(data_dict['X_train']),
                                          torch.Tensor(data_dict['y_train']))
            test_dataset = TensorDataset(torch.Tensor(data_dict['X_test']),
                                         torch.Tensor(data_dict['y_test']))
        else:
            raise ValueError('Invalid data_source')


        # Limit the training samples :
        if limit_train_samples:  # if not none/zero
            train_dataset = reduce_train_set(train_dataset, limit_train_samples)

        # Create data loaders:
        kwargs = {'num_workers': 0, 'pin_memory': True}

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
    transform += [transforms.Normalize((0.5 ,0.5 ,0.5), (0.5 ,0.5 ,0.5))]  # transform to [-1,1]

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
        info = {'input_shape': (1, 28, 28),  'n_classes': 10, 'type': 'multi_class'}

    elif prm.data_source == 'CIFAR10':
        info = {'input_shape': (3, 32, 32), 'n_classes': 10, 'type': 'multi_class'}

    elif prm.data_source == 'Omniglot':
        info = {'input_shape': (1, 28, 28), 'n_classes': prm.N_Way, 'type': 'multi_class'}

    elif prm.data_source == 'SmallImageNet':
        info = {'input_shape': (3, 84, 84), 'n_classes': prm.N_Way, 'type': 'multi_class'}

    elif prm.data_source == 'binarized_MNIST':
        info = {'input_shape': (1, 28, 28),  'n_classes': 2, 'type': 'binary_class'}
        # note: since we have two classes, we can have one output

    elif prm.data_source in BINARY_DATASETS:
        # See https://papers.nips.cc/paper/2021/file/0415740eaa4d9decbc8da001d3fd805f-Supplemental.pdf
        info = {'n_classes': 2, 'type': 'binary_class'}
        # note: since we have two classes, we can have one output

    else:
        raise ValueError('Invalid data_source')

    if info['type'] == 'multi_class':
        # label is the argmax of the output vector
        info['output_dim'] =  info['n_classes']
    elif info['type'] == 'binary_class':
        # label are -/+1
        info['output_dim'] = 1

    return info




# -------------------------------------------------------------------------------------------
#  Batch extraction
# -------------------------------------------------------------------------------------------

def get_batch_vars(batch_data, prm):
    ''' Transform batch to variables '''
    inputs, targets = batch_data
    inputs, targets = inputs.to(prm.device), targets.to(prm.device)
    return inputs, targets


def get_next_batch_cyclic(data_iterator, data_generator):
    ''' get sample from iterator, if it finishes then restart  '''
    try:
        batch_data = data_iterator.next()
    except StopIteration:
        # in case some task has less samples - just restart the iterator and re-use the samples
        data_iterator = iter(data_generator)
        batch_data = data_iterator.next()
    return batch_data

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

def create_limited_pixel_permute_trans(prm):
    info = get_info(prm)
    input_shape = info['input_shape']
    input_size = input_shape[0] * input_shape[1] * input_shape[2]
    inds_permute = torch.LongTensor(np.arange(0, input_size))

    for i_shuffle in range(prm.n_pixels_shuffles):
        i1 = np.random.randint(0, input_size)
        i2 = np.random.randint(0, input_size)
        temp = inds_permute[i1]
        inds_permute[i1] = inds_permute[i2]
        inds_permute[i2] = temp

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
    # debug: show  image
    # import matplotlib.pyplot as plt
    # plt.imshow(x.numpy()[0])
    # plt.show()

    return x

def create_label_permute_trans(prm):
    info = get_info(prm)
    inds_permute = torch.randperm(info['n_classes'])
    transform_func = lambda target: inds_permute[target]
    return transform_func


def create_label_binarize(prm, thresh):
    # binarizes the labels (0 or 1)
    info = get_info(prm)
    transform_func = lambda target: (target >= thresh)
    return transform_func



def create_rotation_trans():
    # all images in task are rotated by some random angle from [0,90,180,270]
    n_rot = np.random.randint(4)
    return lambda x: rotate_im(x, n_rot)

def rotate_im(x, n_rot):
    x = torch.from_numpy(np.rot90(x.squeeze().numpy(), n_rot).copy()).unsqueeze_(0)
    # show  image
    # import matplotlib.pyplot as plt
    # plt.imshow(x.numpy()[0])
    # plt.show()
    return x

def reduce_train_set(train_dataset, limit_train_samples):
    # Limit the training samples :
    n_train_samples_orig = len(train_dataset)
    if limit_train_samples and limit_train_samples < n_train_samples_orig:

        indices = torch.randperm(n_train_samples_orig)[:limit_train_samples]
        train_dataset = data_utils.Subset(train_dataset, indices)
        print(f'Dataset reduced to {len(train_dataset)} samples')

        # if isinstance(train_dataset.data, np.ndarray):
        #     sampled_inds = np.random.permutation(n_train_samples_orig)[:limit_train_samples]
        #     train_dataset.data = train_dataset.data[sampled_inds]
        #     train_dataset.labels = np.array(train_dataset.labels)[sampled_inds]
        # else:
        #     sampled_inds = torch.randperm(n_train_samples_orig)[:limit_train_samples]
        #     train_dataset = torch.utils.data.Subset(train_dataset, sampled_inds)
        #     # train_dataset.train_data = train_dataset.train_data[sampled_inds]
        #     # train_dataset.train_labels = train_dataset.train_labels[sampled_inds]
    return train_dataset
