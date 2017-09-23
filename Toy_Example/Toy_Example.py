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
#  Standard Learning
# -------------------------------------------------------------------------------------------

def standard_learn():

    # Init weights:
    w = Variable(torch.randn(n_tasks, n_dim).cuda(), requires_grad=True)

    learning_rate = 1e-1

    # create your optimizer
    optimizer = optim.Adam([w], lr=learning_rate)

    n_epochs = 300
    batch_size = 128

    for i_epoch in range(n_epochs):

        # Sample data:
        b_task = np.random.randint(0, n_tasks)  # get batch from a random task
        batch_size_curr = min(n_samples_list[b_task], batch_size)
        batch_inds = np.random.choice(n_samples_list[b_task], batch_size_curr)
        task_data = torch.from_numpy(data_set[b_task][batch_inds])
        task_data = Variable(task_data.cuda(), requires_grad=False)

        # Loss:
        loss = (w[b_task] - task_data).pow(2).mean()

        # Gradient step:
        optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        optimizer.step()  # Does the update

        if i_epoch % 100 == 0:
            print('Step: {0}, loss: {1}'.format(i_epoch, loss.data[0]))

    # Switch learned parameters back to numpy:
    w = w.data.cpu().numpy()

    #  Plots:
    fig1 = plt.figure()
    ax = plt.subplot(0, aspect='equal')
    for i_task in range(n_tasks):
        plt.plot(data_set[i_task][:, 0], data_set[i_task][:, 1], '.',
                 label='Task {0}'.format(i_task))
        plt.plot(w[i_task][0], w[i_task][1], 'x', label='Learned w in task {0}'.format(i_task))
    plt.legend()



# -------------------------------------------------------------------------------------------
#  PAC-Bayesian meta-learning
# -------------------------------------------------------------------------------------------

def bayes_learn():

    # Define prior:
    learn_prior_flag = True
    w_P_mu = Variable(torch.zeros(n_dim).cuda(), requires_grad=False)
    w_P_log_sigma = Variable(torch.zeros(n_dim).cuda(), requires_grad=False)
    sigma_sqr_prior = torch.exp(2 * w_P_log_sigma)

    # Init posteriors:
    w_mu = Variable(torch.randn(n_tasks, n_dim).cuda(), requires_grad=True)
    w_log_sigma = Variable(torch.randn(n_tasks, n_dim).cuda(), requires_grad=True)


    learning_rate = 1e-1

    # create your optimizer
    optimizer = optim.Adam([w_mu, w_log_sigma], lr=learning_rate)

    n_epochs = 500
    batch_size = 128

    for i_epoch in range(n_epochs):

        # Sample data:
        b_task = np.random.randint(0, n_tasks)  # get batch from a random task
        batch_size_curr = min(n_samples_list[b_task], batch_size)
        batch_inds = np.random.choice(n_samples_list[b_task], batch_size_curr)
        task_data = torch.from_numpy(data_set[b_task][batch_inds])
        task_data = Variable(task_data.cuda(), requires_grad=False)

        # Re-Parametrization:
        w_sigma = torch.exp(w_log_sigma[b_task])
        epsilon = Variable(torch.randn(n_dim).cuda(), requires_grad=False)
        w = w_mu[b_task] + w_sigma * epsilon

        # Empirical Loss:
        empirical_loss = (w - task_data).pow(2).mean()

        # Complexity terms:
        complex_term = 0
        for i_task in range(n_tasks):
            p = 1e-9  # add small positive number to avoid division by zero due to numerical errors
            sigma_sqr_post = torch.exp(2 * w_log_sigma[i_task])
            kl_dist = torch.sum(w_P_log_sigma - w_log_sigma[i_task] +
                                     ((w_mu[i_task] - w_P_mu).pow(2) + sigma_sqr_post) / (2 * sigma_sqr_prior + p) - 0.5)
            n_samples = n_samples_list[i_task]
            # Complexity by 'PAC_Bayes_McAllaster':
            # delta = 0.95
            # complex_term += torch.sqrt((1 / (2 * n_samples)) *
            #                            (kl_dist + np.log(2 * np.sqrt(n_samples) / delta)))
            # Variational Bayes:
            complex_term += (1 / n_samples) * kl_dist


        # Total objective:
        objective = empirical_loss + complex_term

        # Gradient step:
        optimizer.zero_grad()  # zero the gradient buffers
        objective.backward()
        optimizer.step()  # Does the update

        if i_epoch % 100 == 0:
            print('Step: {0}, objective: {1}'.format(i_epoch, objective.data[0]))

    # Switch  back to numpy:
    w_mu = w_mu.data.cpu().numpy()
    w_log_sigma = w_log_sigma.data.cpu().numpy()
    w_sigma = np.exp(w_log_sigma)
    w_P_mu = w_P_mu.data.cpu().numpy()
    w_P_log_sigma = w_P_log_sigma.data.cpu().numpy()
    w_P_sigma =  np.exp(w_P_log_sigma)

    #  Plots:
    fig1 = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    # plot prior:
    plt.plot(w_P_mu[0], w_P_mu[1], 'o', label='prior mean ')
    ell = Ellipse(xy=(w_P_mu[0], w_P_mu[1]),
                  width=w_P_sigma[0], height=w_P_sigma[1],
                  angle=0, color='blue')
    ell.set_facecolor('none')
    ax.add_artist(ell)
    for i_task in range(n_tasks):
        # plot task data points:
        plt.plot(data_set[i_task][:, 0], data_set[i_task][:, 1], '.',
                 label='Task {0}'.format(i_task))
        # plot posterior:
        plt.plot(w_mu[i_task][0], w_mu[i_task][1], 'o', label='posterior mean {0}'.format(i_task))
        ell = Ellipse(xy=(w_mu[i_task][0], w_mu[i_task][1]),
                      width=w_sigma[i_task][0], height=w_sigma[i_task][1],
                      angle=0, color='black')
        ell.set_facecolor('none')
        ax.add_artist(ell)



    plt.legend()

# -------------------------------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------------------------------
# standard_learn()
bayes_learn()

plt.show()