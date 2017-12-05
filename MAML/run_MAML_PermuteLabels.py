from subprocess import call

call(['python', 'main_MAML.py',
      '--data-source', 'MNIST',
      '--data-transform', 'Permute_Labels',
      '--model-name', 'ConvNet3',
      # MAML hyper-parameters:
      '--alpha', '0.1',
      '--n_meta_train_grad_steps', '2',
      '--n_meta_train_epochs', '300',
      '--n_meta_test_epochs', '100',
      ])
