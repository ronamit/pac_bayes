from subprocess import call
import os
import timeit, time

# Runs meta-training with the specified hyper-parameters
# and meta-testing with a range of 'meta-test gradient steps'

alpha = 0.01
n_meta_train_grad_steps = 2

base_run_name = 'Labels_Alpha_{}_TrainGrads_{}'.format(alpha, n_meta_train_grad_steps)
base_run_name = base_run_name.replace('.','_')


start_time = timeit.default_timer()

for n_meta_test_grad_steps in range(1, 20):
    if n_meta_test_grad_steps == 1:
        mode = 'MetaTrain'
    else:
        mode = 'LoadMetaModel'

    call(['python', 'main_MAML.py',
          '--run-name', base_run_name + '/' + 'TestGrads_' + str(n_meta_test_grad_steps),
          '--mode', mode,
          '--data-source', 'MNIST',
          '--n_train_tasks', '5',
          '--data-transform', 'Permute_Labels',
          '--model-name', 'ConvNet3',
          # MAML hyper-parameters:
          '--alpha', str(alpha),
          '--n_meta_train_grad_steps', str(n_meta_train_grad_steps),
          '--n_meta_train_iterations', '300', #  '300',
          '--meta_batch_size', '32',
          '--n_meta_test_grad_steps', str(n_meta_test_grad_steps),
          '--n_test_tasks', '20',  #  '20',
          '--limit_train_samples_in_test_tasks', '2000',
          ])

stop_time = timeit.default_timer()
print('Total runtime: ' +
      time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)))