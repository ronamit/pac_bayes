from subprocess import call
import argparse
import os

# Select GPU to run:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

n_train_tasks = 0  # 0 = infinite

parser = argparse.ArgumentParser()

parser.add_argument('--complexity_type', type=str,
                    help=" The learning objective complexity type",
                    default='NewBoundMcAllaster')
# 'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina'   NewBoundMcAllaster / NewBoundSeeger'"


args = parser.parse_args()

complexity_type = args.complexity_type

call(['python', 'main_Meta_Bayes.py',
      '--run-name', 'SmallTasks_SmallImageNet',
      '--data-source', 'SmallImageNet',
      '--N_Way', '5',
      '--K_Shot_MetaTrain', '1000',
      '--K_Shot_MetaTest', '500',
      '--data-transform', 'None',
      '--limit_train_samples_in_test_tasks', '0',
      '--n_train_tasks',  str(n_train_tasks),
      '--mode', 'MetaTrain',
      '--complexity_type',  complexity_type,
      '--model-name', 'OmConvNet',  ##  OmConvNet
      '--n_meta_train_epochs', '300',  # 150
      '--n_inner_steps', '500',  # 5000
      '--n_meta_test_epochs', '300',
      '--n_test_tasks', '20',
      '--meta_batch_size', '5',
      ])
