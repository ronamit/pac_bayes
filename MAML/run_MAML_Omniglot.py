from subprocess import call

call(['python', 'main_MAML.py',
      '--data-source', 'Omniglot',
      '--n_train_tasks', '64',
      '--data-transform', 'None',
      '--model-name', 'OmConvNet',
      # MAML hyper-parameters:
      '--alpha', '0.4',
      '--n_meta_train_grad_steps', '1',
      '--n_meta_train_iterations', '60000',
      '--meta_batch_size', '32',
      '--n_meta_test_grad_steps', '100',
      ])