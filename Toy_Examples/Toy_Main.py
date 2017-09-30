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


# -------------------------------------------------------------------------------------------
# Define scenario
# -------------------------------------------------------------------------------------------

n_dim = 2

data_type = 1 # 0 \ 1

if data_type == 0:
    n_tasks = 2
    # number of samples in each task:
    n_samples_list =[10, 200]
    # True means vector for each task [n_dim x n_tasks]:
    true_mu = [[-1.0,-1.0], [+1.0, +1.0]]
    # True sigma vector for each task [n_dim x n_tasks]:
    true_sigma = [[0.1,0.1], [0.1, 0.1]]

elif data_type == 1:
    n_tasks = 2
    # number of samples in each task:
    n_samples_list =[100, 100]
    # True means vector for each task [n_dim x n_tasks]:
    true_mu = [[4, 2], [8, 2]]
    # True sigma vector for each task [n_dim x n_tasks]:
    true_sigma = [[0.3, 0.3], [0.5, 0.5]]

else:
    raise ValueError('Invalid data_type')


# -------------------------------------------------------------------------------------------
#  Generate data samples
# -------------------------------------------------------------------------------------------
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
learning_type = 'MetaLearnWeights' # 'Standard' \ 'Bayes_FixedPrior' \ 'MetaLearnPosteriors' \ MetaLearnWeights
# 'Standard' = Learn optimal weights in each task separately
# 'Bayes_FixedPrior' = Learn posteriors for each task, assuming a fixed shared prior
# 'MetaLearnPosteriors' = Learn weights for each task and the shared prior jointly
#

if learning_type == 'Standard':
    import toy_standard
    toy_standard.learn(data_set)

if learning_type == 'Bayes_FixedPrior':
    import toy_Bayes_FixedPrior
    toy_Bayes_FixedPrior.learn(data_set)

if learning_type == 'MetaLearnPosteriors':
    import toy_MetaLearnPosteriors
    complexity_type = 'PAC_Bayes_McAllaster' # 'PAC_Bayes_McAllaster' \ 'Variational_Bayes' \ 'KL'
    toy_MetaLearnPosteriors.learn(data_set, complexity_type)

if learning_type == 'MetaLearnWeights':
    import toy_MetaLearnWeights
    complexity_type = 'Variational_Bayes'
    toy_MetaLearnWeights.learn(data_set, complexity_type)

plt.show()