from subprocess import call

call(['python', 'main_Meta_Bayes.py',
      '--data-source', 'MNIST',
      '--data-transform', 'Permute_Pixels',
      '--n_train_tasks', '10',
      '--model-name',   'FcNet3',
      '--complexity_type',  'NewBoundMcAllaster', #'NewBoundMcAllaster',
      '--n_test_tasks', '10',
      '--n_meta_train_epochs', '1000',
      '--n_meta_test_epochs', '1000',  # 300
      '--meta_batch_size', '16',  # 32
      '--mode', 'MetaTrain',  # 'MetaTrain'  \ 'LoadMetaModel'
      '--limit_train_samples_in_test_tasks', '2000',
      # '--override_eps_std', '1e-3',
      ])


