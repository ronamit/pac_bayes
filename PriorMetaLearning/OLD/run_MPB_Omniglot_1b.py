from subprocess import call
import os

# Select GPU to run:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


call(['python', 'main_Meta_Bayes.py',
      '--run-name', 'Omniglot_1b',
      '--data-source', 'Omniglot',  # MNIST Omniglot
      '--data-transform', 'Rotate90',
      '--N_Way', '5',
      '--K_Shot_MetaTrain', '20',
      '--K_Shot_MetaTest', '5',
      '--n_train_tasks', '100',
      '--model-name',   'OmConvNet',
      '--complexity_type', 'McAllaster',
      '--n_test_tasks', '100',
      '--n_meta_train_epochs', '2000', # 3000   # 500 = 1 hour
      '--n_inner_steps', '100',
      '--meta_batch_size', '4',  # 32
      '--mode', 'MetaTrain',  # 'MetaTrain'  \ 'LoadMetaModel'
      # '--override_eps_std', '1.0',
      ])


# Results:
# K_Shot_MetaTrain', '5', --K_Shot_MetaTest', '5',
#   --n_meta_train_epochs', '3000', 6 hours, Avg test err: 12.6%, STD: 7.45%
#   --n_meta_train_epochs', '10000 20 hours, ', Avg test err: 13.7%, STD: 8.04%


# Results - limited tasks:
# '--n_train_tasks', '20', --n_meta_train_epochs', '10000',  01 hours, Meta-Testing - Avg test err: 67.0%, STD: 13.6%



# '--n_train_tasks', '50', '--n_meta_train_epochs', '1000' 29 minutes - Meta-Testing - Avg test err: 35.2%, STD: 12.0%


# '--n_train_tasks', '100', '--n_meta_train_epochs', '2000'   '--K_Shot_MetaTrain', '5'  23 hours - Meta-Testing - - Avg test err: 10.6%, STD: 6.75%


# '--n_train_tasks', '100', '--n_meta_train_epochs', '2000'   '--K_Shot_MetaTrain', '20' --04 hours,  Meta-Testing - Avg test err: 24.4%, STD: 11.6%