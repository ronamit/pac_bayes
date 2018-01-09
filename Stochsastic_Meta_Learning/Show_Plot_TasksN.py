

import pickle, os
import numpy as np



dir_path = './saved'

# -------------------------------------------------------------------------------------------
#  Print results
# -------------------------------------------------------------------------------------------



# Getting back the objects:
with open(os.path.join(dir_path, 'results.pkl'), 'rb') as f:  # Python 3: open(..., 'rb')
    mean_error_per_tasks_n, std_error_per_tasks_n, n_tasks_vec = pickle.load(f)


start_n = 1
n_tasks_vec = n_tasks_vec[start_n:]
mean_error_per_tasks_n = mean_error_per_tasks_n[start_n:]
std_error_per_tasks_n = std_error_per_tasks_n[start_n:]


import matplotlib.pyplot as plt
plt.figure()

plt.errorbar(n_tasks_vec, 100*mean_error_per_tasks_n, yerr=100*std_error_per_tasks_n)
plt.xticks(n_tasks_vec)
plt.xlabel('Number of training-tasks')
plt.ylabel('Error on new task [%]')
plt.show()
