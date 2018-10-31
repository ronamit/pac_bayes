from subprocess import call
import argparse
import os

# Select GPU to run:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

n_train_tasks = 10  # 0 = infinite

parser = argparse.ArgumentParser()

parser.add_argument('--complexity_type', type=str,
                    help=" The learning objective complexity type",
                    default='McAllaster')
# 'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina'   McAllaster / Seeger'"


args = parser.parse_args()

complexity_type = args.complexity_type

call(['python', 'main_Meta_Bayes.py',
      '--run-name', 'SmallImageNet_0',
      '--data-source', 'SmallImageNet',
      '--N_Way', '5',
      '--K_Shot_MetaTrain', '1000',
      '--K_Shot_MetaTest', '500',
      '--data-transform', 'None',
      '--limit_train_samples_in_test_tasks', '0',
      '--n_train_tasks',  str(n_train_tasks),
      '--mode', 'MetaTrain',
      '--complexity_type',  complexity_type,
      '--model-name', 'OmConvNet32',
      '--n_meta_train_epochs', '500',  # 150
      # '--n_inner_steps', '5000',  # 5000
      '--n_meta_test_epochs', '500',
      '--n_test_tasks', '10',
      '--meta_batch_size', '5',
      ])

#  n_train_tasks = 30 , 04 hours, Avg test err: 39.7%, STD: 7.56%
#  n_train_tasks = 100,  11 hours,  37.3%, STD: 7.66% (standard is 19.5%)

# n_train_tasks = 10  05 hours Avg test err: 39.8%, STD: 8.32%