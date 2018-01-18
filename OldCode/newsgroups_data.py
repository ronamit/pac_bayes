
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from pprint import pprint
import random
import numpy as np

n_labels_total = 20


class newsgroups_dataset():
    def __init__(self):
        super(newsgroups_dataset, self).__init__()
        newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        newsgroups_test = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        vectorizer = TfidfVectorizer()
        self.train_vectors = vectorizer.fit_transform(newsgroups_train.data)
        self.train_targets = newsgroups_train.target
        self.test_vectors = vectorizer.fit_transform(newsgroups_test.data)
        self.test_targets = newsgroups_test.target



# def __getitem__(self, index):

    # def __len__(self):






def create_meta_split(n_meta_train_labels):
    all_labels = list(range(n_labels_total))
    labels_splits = {}
    # Take random n_meta_train_labels chars as meta-train and rest as meta-test
    random.shuffle(all_labels)
    labels_splits['meta_train'] = all_labels[:n_meta_train_labels]
    labels_splits['meta_test'] = all_labels[n_meta_train_labels:]
    return labels_splits



def get_task(labels_in_split,  n_labels, k_train_shot):
    # labels_list = labels of current split
    newsgroups_train = fetch_20newsgroups(subset='train')
    all_categories = list(newsgroups_train.target_names)

    # Draw n_labels
    n_labels_in_split = len(labels_in_split)
    label_inds = np.random.choice(n_labels_in_split, n_labels, replace=False)
    task_labels = [labels_in_split[ind] for ind in label_inds]
    task_categories = [all_categories[label] for label in task_labels]

    train_dataset = fetch_20newsgroups(subset='train', categories=task_categories, shuffle=True)
    test_dataset = fetch_20newsgroups(subset='test', categories=task_categories, shuffle=True)

    # Take subset of training data:
    k_train_shot


    return train_dataset, test_dataset





dataset = newsgroups_dataset()

n_meta_train_labels = 10
labels_splits = create_meta_split(n_meta_train_labels)

pprint(labels_splits)

labels_in_split = labels_splits['meta_train']
k_train_shot = 5
n_labels = 3
get_task(labels_in_split, n_labels, k_train_shot)



