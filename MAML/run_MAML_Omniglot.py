from subprocess import call

call(['python', 'main_MAML.py',
      '--data-source', 'Omniglot',
      '--data-transform', 'None',
      '--model-name', 'OmConvNet',
      # MAML hyper-parameters:
      '--alpha', '0.1',
      '--n_meta_train_grad_steps', '2',
      '--n_meta_train_epochs', '300',
      '--meta_batch_size', '32',
      '--n_meta_test_epochs', '100',
      ])
