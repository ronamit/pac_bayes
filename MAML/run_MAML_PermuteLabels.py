from subprocess import call

call(['python', 'main_MAML.py',
      '--data-source', 'MNIST',
      '--n_train_tasks', '5',
      '--data-transform', 'Permute_Labels',
      '--model-name', 'ConvNet3',
      # MAML hyper-parameters:
      '--alpha', '0.01',
      '--n_meta_train_grad_steps', '3',
      '--n_meta_train_iterations', '300',
      '--meta_batch_size', '32',
      '--n_meta_test_grad_steps', '100',
      '--n_test_tasks', '10',
      '--limit_train_samples_in_test_tasks', '2000',
      # '--mode', 'LoadMetaModel',
      # '--meta_model_file_name', 'aaa',
      ])
