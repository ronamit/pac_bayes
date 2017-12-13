from subprocess import call

call(['python', 'main_Meta_Bayes.py',
      '--data-source', 'MNIST',
      '--data-transform', 'Permute_Pixels',
      '--model-name', 'FcNet3'
      '--n_test_tasks', '10',
      '--limit_train_samples_in_test_tasks', '2000',
      ])
