
from __future__ import absolute_import, division, print_function

import timeit
import data_gen

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import random
import common as cmn
from common import count_correct, get_param_from_model, grad_step
from models_standard import get_model


def run_test_max_posterior(model, test_loader, loss_criterion, prm):
    model.eval()
    test_loss = 0
    n_correct = 0
    for batch_data in test_loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm, is_test=True)
        eps_std = 0.0  # test with max-posterior
        outputs = model(inputs, eps_std)
        test_loss += loss_criterion(outputs, targets)  # sum the mean loss in batch
        n_correct += count_correct(outputs, targets)

    n_test_samples = len(test_loader.dataset)
    n_test_batches = len(test_loader)
    test_loss /= n_test_batches
    test_acc = n_correct / n_test_samples
    print('\nMax-Posterior, Test set: Average loss: {:.4}, Accuracy: {:.3} ( {}/{})\n'.format(
        test_loss.data[0], test_acc, n_correct, n_test_samples))
    return test_acc


def run_test_majority_vote(model, test_loader, prm, n_votes=5):
# TODO:  more efficent
    model.eval()
    test_loss = 0
    n_correct = 0
    for batch_data in test_loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm, is_test=True)

        batch_size = prm.test_batch_size
        n_labels = model.out_size
        votes = cmn.zeros_gpu((batch_size, n_labels))
        for i_vote in range(n_votes):
            outputs = model(inputs, eps_std=1.0)
            pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max output
            for i_sample in range(batch_size):
                votes[i_sample, pred[i_sample].cpu().numpy()[0]] += 1

        majority_pred = votes.max(1, keepdim=True)[1]
        n_correct += majority_pred.eq(targets.data.view_as(majority_pred)).cpu().sum()

    n_test_samples = len(test_loader.dataset)
    n_test_batches = len(test_loader)
    test_loss /= n_test_batches
    test_acc = n_correct / n_test_samples
    print('\nMajority-Vote, Test set: Accuracy: {:.3} ( {}/{})\n'.format(
        test_acc, n_correct, n_test_samples))
    return test_acc

###

def get_eps_std(i_epoch, batch_idx, n_meta_batches, prm):

    total_iter = prm.num_epochs * n_meta_batches
    n_iter_stage_1 = int(total_iter * prm.stage_1_ratio)
    n_iter_stage_2 = total_iter - n_iter_stage_1
    n_iter_with_full_eps_std = int(n_iter_stage_2 * prm.full_eps_ratio_in_stage_2)
    full_eps_std = 1.0

    # We gradually increase epsilon's STD from 0 to 1.
    # The reason is that using 1 from the start results in high variance gradients.
    iter_idx = i_epoch * n_meta_batches + batch_idx
    if iter_idx >= n_iter_stage_1:
        eps_std = full_eps_std * (iter_idx - n_iter_stage_1) / (n_iter_stage_2 - n_iter_with_full_eps_std)
    else:
        eps_std = 0.0
    eps_std = min(max(eps_std, 0.0), 1.0)  # keep in [0,1]
    return eps_std


#  -------------------------------------------------------------------------------------------
#  Intra-task complexity for posterior distribution
# -------------------------------------------------------------------------------------------

def get_posterior_complexity_term(complexity_type, prior_model, post_model, n_samples):
    
    kld = get_total_kld(prior_model, post_model)

    if complexity_type == 'KLD':
        complex_term = kld

    elif complexity_type == 'PAC_Bayes_McAllaster':
        delta = 0.95
        complex_term = torch.sqrt((1 / (2 * n_samples)) * (kld))
        # delta = 0.95
        # complex_term = torch.sqrt((1 / (2 * n_samples)) * (kld + np.log(2*np.sqrt(n_samples) / delta))) - \
        #                torch.sqrt((1 / (2 * n_samples)) * (np.log(2*np.sqrt(n_samples) / delta)))
        # I subtracted a const so that the optimization could reach 0

    elif complexity_type == 'PAC_Bayes_Pentina':
        complex_term = np.sqrt(1 / n_samples) * kld

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

    prior_layers_list = list(prior_model.children())
    post_layers_list = list(post_model.children())

    total_kld = 0
    for i_layer, prior_layer in enumerate(prior_layers_list):
        post_layer = post_layers_list[i_layer]

        total_kld += kld_element(post_layer.w, prior_layer.w)
        total_kld += kld_element(post_layer.b, prior_layer.b)

    
    total_kld += 1e-10 # to avoid negative KLD

    return total_kld


def kld_element(post, prior):
    """KL divergence D_{KL}[post(x)||prior(x)] for a fully factorized Gaussian"""

    small_num = 1e-10  # add small positive number to avoid division by zero due to numerical errors

    post_var = torch.exp(post['log_var'])
    prior_var = torch.exp(prior['log_var'])
    #  TODO: maybe the exp can be done once for the KL and forward pass operations for efficiency

    numerator = (post['mean'] - prior['mean']).pow(2) + post_var
    denominator = prior_var + small_num
    kld = 0.5 * torch.sum(prior['log_var'] - post['log_var'] + numerator / denominator - 1)


    return kld
