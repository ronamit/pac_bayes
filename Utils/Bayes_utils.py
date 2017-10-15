
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable
import math
from Utils import common as cmn, data_gen
from Utils.common import count_correct
import torch.nn.functional as F
from Models.layers import StochasticLayer

def run_test_Bayes(model, test_loader, loss_criterion, prm):
    if prm.test_type == 'MaxPosterior':
        return run_test_max_posterior(model, test_loader, loss_criterion, prm)
    elif prm.test_type == 'MajorityVote':
        return run_test_majority_vote(model, test_loader, loss_criterion, prm, n_votes=5)
    elif prm.test_type == 'AvgVote':
        return run_test_avg_vote(model, test_loader, loss_criterion, prm, n_votes=5)
    else:
        raise ValueError('Invalid test_type')


def run_test_max_posterior(model, test_loader, loss_criterion, prm):

    n_test_samples = len(test_loader.dataset)
    n_test_batches = len(test_loader)

    model.eval()
    test_loss = 0
    n_correct = 0
    for batch_data in test_loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm, is_test=True)
        eps_std = 0.0  # test with max-posterior
        outputs = model(inputs, eps_std)
        test_loss += loss_criterion(outputs, targets)  # sum the mean loss in batch
        n_correct += count_correct(outputs, targets)

    test_loss /= n_test_samples
    test_acc = n_correct / n_test_samples
    print('\nMax-Posterior, Test set: Average loss: {:.4}, Accuracy: {:.3} ( {}/{})\n'.format(
        test_loss.data[0], test_acc, n_correct, n_test_samples))
    return test_acc, test_loss.data[0]


def run_test_majority_vote(model, test_loader, loss_criterion, prm, n_votes=9):
#
    n_test_samples = len(test_loader.dataset)
    n_test_batches = len(test_loader)
    model.eval()
    test_loss = 0
    n_correct = 0
    for batch_data in test_loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm, is_test=True)

        batch_size = min(prm.test_batch_size, n_test_samples)
        info = data_gen.get_info(prm)
        n_labels = info['n_classes']
        votes = cmn.zeros_gpu((batch_size, n_labels))
        for i_vote in range(n_votes):
            eps_std = 1.0
            outputs = model(inputs, eps_std)
            test_loss += loss_criterion(outputs, targets)
            pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max output
            for i_sample in range(batch_size):
                pred_val = pred[i_sample].cpu().numpy()[0]
                votes[i_sample, pred_val] += 1

        majority_pred = votes.max(1, keepdim=True)[1]
        n_correct += majority_pred.eq(targets.data.view_as(majority_pred)).cpu().sum()


    test_loss /= n_test_samples
    test_acc = n_correct / n_test_samples
    print('\nMajority-Vote, Test set: Accuracy: {:.3} ( {}/{})\n'.format(
        test_acc, n_correct, n_test_samples))
    return test_acc, test_loss

def run_test_avg_vote(model, test_loader, loss_criterion, prm, n_votes=5):

    n_test_samples = len(test_loader.dataset)
    n_test_batches = len(test_loader)
    model.eval()
    test_loss = 0
    n_correct = 0
    for batch_data in test_loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm, is_test=True)

        batch_size = min(prm.test_batch_size, n_test_samples)
        info = data_gen.get_info(prm)
        n_labels = info['n_classes']
        votes = cmn.zeros_gpu((batch_size, n_labels))
        for i_vote in range(n_votes):
            eps_std = 1.0
            outputs = model(inputs, eps_std)
            test_loss += loss_criterion(outputs, targets)
            votes += outputs.data

        majority_pred = votes.max(1, keepdim=True)[1]
        n_correct += majority_pred.eq(targets.data.view_as(majority_pred)).cpu().sum()

    test_loss /= n_test_samples
    test_acc = n_correct / n_test_samples
    print('\nAveraged-Vote, Test set: Accuracy: {:.3} ( {}/{})\n'.format(
        test_acc, n_correct, n_test_samples))
    return test_acc, test_loss


def get_eps_std(i_epoch, batch_idx, n_meta_batches, prm):

    total_iter = prm.num_epochs * n_meta_batches
    n_iter_stage_1 = int(total_iter * prm.stage_1_ratio)
    n_iter_stage_2 = total_iter - n_iter_stage_1
    n_iter_with_full_eps_std = int(n_iter_stage_2 * prm.full_eps_ratio_in_stage_2)
    full_eps_std = 1.0

    # We gradually increase epsilon'PermuteLabels_Seeger.out STD from 0 to 1.
    # The reason is that using 1 from the start results in high variance gradients.
    iter_idx = i_epoch * n_meta_batches + batch_idx
    if iter_idx >= n_iter_stage_1:
        if n_iter_stage_2 == n_iter_with_full_eps_std:
            eps_std = 1.0
        else:
            eps_std = full_eps_std * (iter_idx - n_iter_stage_1) / (n_iter_stage_2 - n_iter_with_full_eps_std)
    else:
        eps_std = 0.0
    eps_std = min(max(eps_std, 0.0), 1.0)  # keep in [0,1]
    return eps_std


#  -------------------------------------------------------------------------------------------
#  Intra-task complexity for posterior distribution
# -------------------------------------------------------------------------------------------

def get_posterior_complexity_term(complexity_type, prior_model, post_model, n_samples, task_empirical_loss):

    kld = get_total_kld(prior_model, post_model)

    if complexity_type == 'KLD':
        complex_term = kld

    elif complexity_type == 'PAC_Bayes_McAllaster':
        delta = 0.99
        complex_term = torch.sqrt((1 / (2 * n_samples)) * (kld + math.log(2*math.sqrt(n_samples) / delta)))

    elif complexity_type == 'PAC_Bayes_Pentina':
        complex_term = math.sqrt(1 / n_samples) * kld

    elif complexity_type == 'PAC_Bayes_Seeger':
        # Seeger complexity is unique since it requires the empirical loss
        # small_num = 1e-9 # to avoid nan due to numerical errors
        delta = 0.99
        # seeger_eps = (1 / n_samples) * (kld + math.log(2 * math.sqrt(n_samples) / delta))
        seeger_eps = (1 / n_samples) * (kld + math.log(2 * math.sqrt(n_samples) / delta))
        sqrt_arg = 2 * seeger_eps * task_empirical_loss
        sqrt_arg = F.relu(sqrt_arg)  # prevent negative values due to numerical errors
        complex_term = 2 * seeger_eps + torch.sqrt(sqrt_arg)



    elif complexity_type == 'Variational_Bayes':
        # Since we approximate the expectation of the likelihood of all samples,
        # we need to multiply by the average_loss by total number of samples
        # Then we normalize the objective by n_samples
        complex_term = (1 / n_samples) * kld

    elif complexity_type == 'NoComplexity':
        complex_term = Variable(cmn.zeros_gpu(1), requires_grad=False)

    else:
        raise ValueError('Invalid complexity_type')

    return complex_term


def get_total_kld(prior_model, post_model):

    prior_layers_list = [layer for layer in prior_model.children() if isinstance(layer, StochasticLayer)]
    post_layers_list =  [layer for layer in post_model.children() if isinstance(layer, StochasticLayer)]

    total_kld = 0
    for i_layer, prior_layer in enumerate(prior_layers_list):
        post_layer = post_layers_list[i_layer]

        total_kld += kld_element(post_layer.w, prior_layer.w)
        total_kld += kld_element(post_layer.b, prior_layer.b)

    return total_kld


def kld_element(post, prior):
    """KL divergence D_{KL}[post(x)||prior(x)] for a fully factorized Gaussian"""

    post_var = torch.exp(post['log_var'])
    prior_var = torch.exp(prior['log_var'])
    #  TODO: maybe the exp can be done once for the KL and forward pass operations for efficiency

    numerator = (post['mean'] - prior['mean']).pow(2) + post_var
    denominator = prior_var
    kld = 0.5 * torch.sum(prior['log_var'] - post['log_var'] + numerator / denominator - 1)

    # note: don't add small number to denominator, since we need to have zero KL when post==prior.

    return kld
