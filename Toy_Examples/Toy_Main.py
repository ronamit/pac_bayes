from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

import torch
from torch.autograd import Variable
import torch.optim as optim



# -------------------------------------------------------------------------------------------
#  Create data
# -------------------------------------------------------------------------------------------

# Random seed:
seed = 0
if not seed == 0:
    torch.manual_seed(seed)

n_dim = 2

n_tasks = 2

# number of samples in each task:
n_samples_list =[10, 200]

# True means vector for each task [n_dim x n_tasks]:
true_mu = [[-1.0,-1.0], [+1.0, +1.0]]

# True sigma vector for each task [n_dim x n_tasks]:
true_sigma = [[0.1,0.1], [0.1, 0.1]]

data_set = []
for i_task in range(n_tasks):
    task_data = np.random.multivariate_normal(
        mean=true_mu[i_task],
        cov=np.diag(true_sigma[i_task]),
        size=n_samples_list[i_task]).astype(np.float32)

    data_set.append(task_data)

# -------------------------------------------------------------------------------------------
#  Learning
# -------------------------------------------------------------------------------------------
learning_type = 'MetaLearn' # 'Standard' \ 'Bayes_FixedPrior' \ 'MetaLearn'
# 'Standard' = Learn optimal weights in each task separately
# 'Bayes_FixedPrior' = Learn posteriors for each task, assuming a fixed shared prior
# 'MetaLearn' = Learn posteriors for each task and the shared prior jointly

if learning_type == 'Standard':
    import toy_standard
    toy_standard.learn(data_set)

if learning_type == 'Bayes_FixedPrior':
    import toy_Bayes_FixedPrior
    toy_Bayes_FixedPrior.learn(data_set)


if learning_type == 'MetaLearn':
    import toy_MetaLearn
    toy_MetaLearn.learn(data_set)


plt.show()