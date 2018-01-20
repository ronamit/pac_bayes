"""
based on:
https://github.com/cbfinn/maml/blob/master/data/miniImagenet/proc_images.py

Step 1:
    Download ilsvrc2012_img_train.tar from
     http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads
     and place in <data_path>/miniImagenet/images

Step 2:  Run the following commands that extract and DELETES THE ORIGINAL tar file:
    go to images dir
    $ tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    $ find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    // Make sure to check the completeness of the decompression, you should have 1,281,167 images in train folder
        # extract validation data
    cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash


Step 3:
  Download train, val, and test csv files from
  https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet
  and place in MiniImageNet dir

Step 4:
run this script which:
1. resizes the images to 84x84
2. creates the 'mini-imagenet' data set

"""

from __future__ import absolute_import, division, print_function
import csv
import glob
import os

from PIL import Image

from Data_Path import get_data_path

input_dir = os.path.join(get_data_path(), 'MiniImageNet')

path_to_images = os.path.join(input_dir, 'images')

all_images = glob.glob(path_to_images + '/*/*')

# Resize images
for i, image_file in enumerate(all_images):
    try:
        im = Image.open(image_file)
        im = im.resize((84, 84), resample=Image.LANCZOS)
        im.save(image_file)
    except:
        print('Failed on ' + image_file)
    if i % 1000 == 0:
        print(i)


# Put in correct directory
for datatype in ['train', 'val', 'test']:
    os.system('mkdir ' + os.path.join(input_dir, datatype))

    with open(os.path.join(input_dir, datatype) + '.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        last_label = ''
        for i, row in enumerate(reader):
            if i == 0:  # skip the headers
                continue
            label = row[1]
            image_name_t = row[0]
            if label != last_label:
                os.system('mkdir ' + os.path.join(input_dir, datatype, label))
                last_label = label
            # Convert file name format:
            image_name = image_name_t[:9] + '_' + str(int(image_name_t[9:17])) + '.JPEG'
            source_path = os.path.join(path_to_images, label, image_name)
            target_path = os.path.join(input_dir, datatype, label, image_name)
            os.system('mv ' + source_path + ' ' + target_path)