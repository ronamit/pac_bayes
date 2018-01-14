from subprocess import call

call(['python', 'main_Meta_Bayes.py',
      '--data-source', 'Omniglot',  # MNIST Omniglot
      '--data-transform', 'Rotate90',
      '--N_Way', '5',
      '--K_Shot', '20',
      '--n_train_tasks', '0',
      '--model-name',   'ConvNet3', # TODO: implement stochastic 'OmConvNet',
      '--complexity_type', 'NewBoundMcAllaster',
      '--n_test_tasks', '10',
      '--n_meta_train_epochs', '300', # 3000
      '--n_inner_steps', '5000',
      '--meta_batch_size', '8',  # 32
      '--mode', 'MetaTrain',  # 'MetaTrain'  \ 'LoadMetaModel'
      '--override_eps_std', '1.0',
      ])


