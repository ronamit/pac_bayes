
from __future__ import absolute_import, division, print_function

import os
import os.path
import errno
import random
from PIL import Image
import torch.utils.data as data
from torchvision import transforms


# Based on code from:
# https://github.com/katerakelly/pytorch-maml/blob/master/src/task.py
# https://github.com/pytorch/vision/pull/46

# data set: https://github.com/brendenlake/omniglot

def get_omniglot_task(data_path, meta_split, n_labels, k_train_shot, final_input_trans=None, target_transform=None):

    '''
    Samples a N-way k-shot learning task (classification to N classes,
     k training samples per class) from the Omniglot dataset.

     - root = data path
     -  n_labels = number of labels (chars) in the task.
     - meta_split =  'meta_train' / 'meta_test - take chars from meta-train or meta-test data sets
     - k_train_shot - sample this many training examples from each char class,
                      rest of the char examples will be in the test set.

      e.g:
    data_loader = get_omniglot_task(prm, meta_split='meta_train', n_labels=5, k_train_shot=10)
    '''
    if meta_split not in ['meta_train', 'meta_test']:
        raise ValueError('Invalid meta_split')

    # Get data:
    root_path = os.path.join(data_path, 'Omniglot')
    splits_dirs  = maybe_download(root_path)
    data_dir = splits_dirs[meta_split]

    # Sample n_labels classes:
    languages = os.listdir(data_dir)
    chars = []
    for lang in languages:
        chars += [os.path.join(lang, x) for x in os.listdir(os.path.join(data_dir, lang))]
    random.shuffle(chars)
    classes_names = chars[:n_labels]

    train_samp = []
    test_samp = []
    train_targets = []
    test_targets = []

    for i_label in range(n_labels):

        class_dir = classes_names[i_label]
        # First get all instances of that class
        all_class_samples = [os.path.join(class_dir, x) for x in os.listdir(os.path.join(data_dir, class_dir))]
        # Sample k_train_shot instances randomly each for train
        random.shuffle(all_class_samples)
        cls_train_samp = all_class_samples[:k_train_shot]
        train_samp += cls_train_samp
        # Rest go to test set:
        cls_test_samp = all_class_samples[k_train_shot+1:]
        test_samp += cls_test_samp

        # Targets \ labels:
        train_targets += [i_label] * len(cls_train_samp)
        test_targets += [i_label] * len(cls_test_samp)


    # Data transformations list:
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    input_transform = [lambda x: FilenameToPILImage(x, data_dir)]
    input_transform +=  [lambda x: x.resize((28,28), resample=Image.LANCZOS)] # to compare to prior papers
    input_transform += [transforms.ToTensor()]
    input_transform += [normalize]
    # input_transform += [lambda x: x.mean(dim=0).unsqueeze_(0)]  # RGB -> gray scale
    # Switch background to 0 and letter to 1:
    # input_transform += [lambda x: 1.0 - x]
    if final_input_trans:
        input_transform += final_input_trans

    # Create the dataset object:
    train_dataset = omniglot_dataset(train_samp, train_targets, input_transform, target_transform)
    test_dataset = omniglot_dataset(test_samp, test_targets, input_transform, target_transform)


    return train_dataset, test_dataset

# -------------------------------------------------------------------------------------------
#  Class definition
# -------------------------------------------------------------------------------------------
class omniglot_dataset(data.Dataset):
    def __init__(self, samples_paths, targets, transform=None, target_transform=None):
        self.all_items = list(zip(samples_paths, targets))
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):

        img = self.all_items[index][0]
        target = self.all_items[index][1]

        if self.transform:
            for trasns in self.transform:
                img = trasns(img)

        if self.target_transform:
            for trasns in self.target_transform:
                target = trasns(target)

        return img, target

    def __len__(self):
        return len(self.all_items)


def FilenameToPILImage(filename, data_dir):
    """
    Load a PIL RGB Image from a filename.
    """
    file_path = os.path.join(data_dir, filename)
    img=Image.open(file_path).convert('RGB')
    # img.save("tmp.png") # debug
    return img

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
    return splits_dirs

# ----------------   Resize images to 28x28
# """
# Usage instructions:
#     First download the omniglot dataset
#     and put the contents of both images_background and images_evaluation in data/omniglot/ (without the root folder)
#     Then, run the following:
#     cd data/
#     cp -r omniglot/* omniglot_resized/
#     cd omniglot_resized/
#     python resize_images.py
# """
# from PIL import Image
# import glob
#
# image_path = '*/*/'
#
# all_images = glob.glob(image_path + '*')
#
# i = 0
#
# for image_file in all_images:
#     im = Image.open(image_file)
#     im = im.resize((28,28), resample=Image.LANCZOS)
#     im.save(image_file)
#     i += 1
#
#     if i % 200 == 0:
#         print(i)
