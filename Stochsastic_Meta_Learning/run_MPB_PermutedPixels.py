from subprocess import call

call(['python', 'main_Meta_Bayes.py',
      '--data-source', 'MNIST',
      '--data-transform', 'Permute_Pixels',
      '--n_train_tasks', '100',
      '--model-name',   'FcNet3',
      '--complexity_type', 'NewBoundMcAllaster',
      '--n_test_tasks', '100',
      '--n_meta_train_epochs', '200',
      '--n_meta_test_epochs', '300',
      '--meta_batch_size', '16',  # 32
      '--mode', 'LoadMetaModel',  # 'MetaTrain'  \ 'LoadMetaModel'
      '--limit_train_samples_in_test_tasks', '20000',
      # '--override_eps_std', '1e-3',
      ])


