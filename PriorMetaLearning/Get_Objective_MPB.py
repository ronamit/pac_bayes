from __future__ import absolute_import, division, print_function

import torch
from Utils import data_gen
from Utils.Bayes_utils import get_task_complexity, get_meta_complexity_term
from Utils.common import net_weights_magnitude, count_correct, get_value, net_weights_dim, zeros_gpu

# -------------------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------------------
def get_objective(prior_model, prm, mb_data_loaders, mb_iterators, mb_posteriors_models, loss_criterion, n_train_tasks):
    '''  Calculate objective based on tasks in meta-batch '''
    # note: it is OK if some tasks appear several times in the meta-batch

    n_tasks_in_mb = len(mb_data_loaders)

    correct_count = 0
    sample_count = 0

    if prm.divergence_type == 'Wasserstein':
        d = net_weights_dim(prior_model)
        hyper_kl = torch.sqrt(net_weights_magnitude(prior_model, p=2) + d * (prm.kappa_prior - prm.kappa_post)**2)

    elif  prm.divergence_type == 'Wasserstein_NoSqrt':
        d = net_weights_dim(prior_model)
        hyper_kl = net_weights_magnitude(prior_model, p=2) + d * (prm.kappa_prior - prm.kappa_post) ** 2

    elif prm.divergence_type == 'KL':
        # KLD between hyper-posterior and hyper-prior:
        hyper_kl = (1 / (2 * prm.kappa_prior**2)) * net_weights_magnitude(prior_model, p=2)
    else:
        raise ValueError('Invalid prm.divergence_type')

    # Hyper-prior term:
    meta_complex_term = get_meta_complexity_term(hyper_kl, prm, n_train_tasks)


    avg_empiric_loss_per_task = zeros_gpu(n_tasks_in_mb)
    complexity_per_task = zeros_gpu(n_tasks_in_mb)
    n_samples_per_task = zeros_gpu(n_tasks_in_mb)# how many sampels there are total in each task (not just in a batch)

    # ----------- loop over tasks in meta-batch -----------------------------------#
    for i_task in range(n_tasks_in_mb):

        n_samples = mb_data_loaders[i_task]['n_train_samples']
        n_samples_per_task[i_task] = n_samples

        # get sample-batch data from current task to calculate the empirical loss estimate:
        batch_data = data_gen.get_next_batch_cyclic(mb_iterators[i_task], mb_data_loaders[i_task]['train'])

        # The posterior model corresponding to the task in the batch:
        post_model = mb_posteriors_models[i_task]
        post_model.train()

        # Monte-Carlo iterations:
        n_MC = prm.n_MC

        # Monte-Carlo loop
        for i_MC in range(n_MC):

            # Debug
            # print(targets[0].data[0])  # print first image label
            # import matplotlib.pyplot as plt
            # plt.imshow(inputs[0].cpu().data[0].numpy())  # show first image
            # plt.show()

            # get batch variables:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)
            # note: we sample new batch in eab MC run to get lower variance estimator
            batch_size = inputs.shape[0]

            # Empirical Loss on current task:
            outputs = post_model(inputs)
            avg_empiric_loss_curr = (1 / batch_size) * loss_criterion(outputs, targets)

            correct_count += count_correct(outputs, targets)
            sample_count += inputs.size(0)

            # Intra-task complexity of current task:
            curr_complexity = get_task_complexity(prm, prior_model, post_model,
                n_samples, avg_empiric_loss_curr, hyper_kl, n_train_tasks=n_train_tasks, noised_prior=True)

            avg_empiric_loss_per_task[i_task] += (1 / n_MC) * avg_empiric_loss_curr
            complexity_per_task[i_task] += (1 / n_MC) * curr_complexity
        # end Monte-Carlo loop

    # end loop over tasks in meta-batch


    # Approximated total objective:
    if prm.complexity_type == 'Variational_Bayes':
        # note that avg_empiric_loss_per_task is estimated by an average over batch samples,
        #  but its weight in the objective should be considered by how many samples there are total in the task
        total_objective = (avg_empiric_loss_per_task * n_samples_per_task + complexity_per_task).mean() * n_train_tasks + meta_complex_term
        # total_objective = ( avg_empiric_loss_per_task * n_samples_per_task + complexity_per_task).mean() + meta_complex_term

    else:
        total_objective = avg_empiric_loss_per_task.mean() + complexity_per_task.mean() + meta_complex_term

    info = {'sample_count': get_value(sample_count), 'correct_count': get_value(correct_count),
                  'avg_empirical_loss': get_value(avg_empiric_loss_per_task.mean()),
                  'avg_intra_task_comp': get_value(complexity_per_task.mean()),
                  'meta_comp': get_value(meta_complex_term)}
    return total_objective, info
