from subprocess import call
import os

# Select GPU to run:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


call(['python', 'main_Meta_Bayes.py',
      '--run-name', 'Omniglot_1',
      '--data-source', 'Omniglot',  # MNIST Omniglot
      '--data-transform', 'Rotate90',
      '--N_Way', '5',
      '--K_Shot_MetaTrain', '20',
      '--K_Shot_MetaTest', '5',
      '--n_train_tasks', '0',
      '--model-name',   'OmConvNet',
      '--complexity_type', 'NewBoundMcAllaster',
      '--n_test_tasks', '100',
      '--n_meta_train_epochs', '10', # 3000
      '--n_inner_steps', '1000',
      '--meta_batch_size', '4',  # 32
      '--mode', 'MetaTrain',  # 'MetaTrain'  \ 'LoadMetaModel'
      # '--override_eps_std', '1.0',
      ])


