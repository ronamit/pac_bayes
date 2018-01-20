from __future__ import absolute_import, division, print_function
import pickle, os
import matplotlib.pyplot as plt



paths_to_result_files = ['Stochsastic_Meta_Learning/saved/TasksN_permuted_Labels.pkl',
                         'Stochsastic_Meta_Learning/saved/Shuffle100Pix.pkl',
                         'Stochsastic_Meta_Learning/saved/Shuffle200Pix.pkl',
                         'Stochsastic_Meta_Learning/saved/Shuffle300Pix.pkl']


legend_names = ['permuted labels', '100 pixel shuffles', '200 pixel shuffles', '300 pixel shuffles']

n_expirements = len(paths_to_result_files)

plt.figure()

for i_exp in range(n_expirements):

    with open(paths_to_result_files[i_exp], 'rb') as f:
        mean_error_per_tasks_n, std_error_per_tasks_n, n_tasks_vec = pickle.load(f)

    plt.errorbar(n_tasks_vec, 100 * mean_error_per_tasks_n, yerr=100 * std_error_per_tasks_n,
                 label=legend_names[i_exp])
    plt.xticks(n_tasks_vec)



plt.legend()
plt.xlabel('Number of training-tasks')
plt.ylabel('Error on new task [%]')
plt.show()
