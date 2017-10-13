
from __future__ import absolute_import, division, print_function
import os
import os.path
import errno
import numpy as np
import random
from PIL import Image
import torch.utils.data as data


# Based on code from:
# https://github.com/katerakelly/pytorch-maml/blob/master/src/task.py
# https://github.com/pytorch/vision/pull/46

# data set: https://github.com/brendenlake/omniglot

def get_omniglot_sets(root, meta_split, n_labels, k_train_shot):

    '''
    Samples a N-way k-shot learning task (classification to N classes,
     k training samples per class) from the Omniglot dataset.

     - root = data path
     -  n_labels = number of labels (chars) in the task.
     - meta_split =  'meta_train' / 'meta_test - take chars from meta-train or meta-test data sets
     - k_train_shot - sample this many training examples from each char class,
                      rest of the char examples will be in the test set.
    '''

    # Get data:
    splits_dirs  = maybe_download(root)
    data_dir = splits_dirs[meta_split]

    # Sample n_labels classes:
    languages = os.listdir(data_dir)
    chars = []
    for lang in languages:
        chars += [os.path.join(lang, x) for x in os.listdir(os.path.join(data_dir, lang))]
    random.shuffle(chars)
    classes = chars[:n_labels]
    labels = np.array(range(len(classes)))
    labels = dict(zip(classes, labels))



# -------------------------------------------------------------------------------------------
#  Auxiliary functions
# -------------------------------------------------------------------------------------------

def check_exists(splits_dirs):
    paths = list(splits_dirs.values())
    return all([os.path.exists(path) for path in paths])

def maybe_download(root):
    from six.moves import urllib
    import zipfile

    splits_dirs = {'meta_train':  os.path.join(root, 'processed', 'images_background'),
                 'meta_test': os.path.join(root, 'processed', 'images_evaluation')}
    if check_exists(splits_dirs):
        return splits_dirs

    # download files
    data_urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip']

    raw_folder = 'raw'
    processed_folder = 'processed'
    try:
        os.makedirs(os.path.join(root, raw_folder))
        os.makedirs(os.path.join(root, processed_folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    for url in data_urls:
        print('== Downloading ' + url)
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(root, raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())
        file_processed = os.path.join(root, processed_folder)
        print("== Unzip from "+file_path+" to "+file_processed)
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(file_processed)
        zip_ref.close()
    print("Download finished.")


