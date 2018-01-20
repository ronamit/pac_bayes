from subprocess import call

n_train_tasks = 5
complexity_type = 'Variational_Bayes'  #  'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina'   NewBoundMcAllaster / NewBoundSeeger'"


call(['python', 'main_Meta_Bayes.py',
      '--run-name', 'PermutedLabels_{}_Tasks_{}_Comp'.format(n_train_tasks, complexity_type),
      '--data-source', 'MNIST',
      '--data-transform', 'Permute_Labels',
      '--limit_train_samples_in_test_tasks', '2000',
      '--n_train_tasks',  str(n_train_tasks),
      '--mode', 'MetaTrain',
      '--complexity_type',  complexity_type,
      '--model-name', 'ConvNet3',
      '--n_meta_train_epochs', '150',
      '--n_meta_test_epochs', '200',
      '--n_test_tasks', '100',

      ])
