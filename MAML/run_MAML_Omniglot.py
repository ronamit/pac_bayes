from subprocess import call

call(['python', 'main_MAML.py',
      '--data-source', 'Omniglot',
      '--data-transform', 'None',
      '--model-name', 'OmConvNet',
      # MAML hyper-parameters:
      '--alpha', '0.4',
      '--n_meta_train_grad_steps', '1',
      '--n_meta_train_epochs', '60000',
      '--meta_batch_size', '32',
      '--n_meta_test_epochs', '3',
      ])
