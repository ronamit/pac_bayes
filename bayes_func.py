
from __future__ import absolute_import, division, print_function

import timeit
import data_gen

import numpy as np
import torch
from torch.autograd import Variable
import random
import common as cmn
from common import count_correct, get_param_from_model, grad_step
from models_standard import get_model

#  -------------------------------------------------------------------------------------------
#  Regularization
# -------------------------------------------------------------------------------------------

def net_L1_norm(model):
    l1_crit = torch.nn.L1Loss(size_average=False)
    total_norm = 0
    for param in model.parameters():
        target = Variable(cmn.zeros_gpu(param.size()), requires_grad=False)  # dummy target
        total_norm += l1_crit(param, target)
    return total_norm


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
    post_layers_list= list(post_model.children())

    total_kld = 0
    for i_layer, prior_layer in enumerate(prior_layers_list):
        post_layer = post_layers_list[i_layer]

        total_kld += kld_element(post_layer.w, prior_layer.w)
        total_kld += kld_element(post_layer.b, prior_layer.b)

    return total_kld


def kld_element(post, prior):

    small_num = 1e-20  # add small positive number to avoid division by zero due to numerical errors

    post_var = torch.exp(post['log_var'])
    prior_var = torch.exp(prior['log_var'])
    #  TODO: maybe the exp can be done once for the KL and forward pass operations for efficiency

    kld = torch.sum(prior['log_var'] - post['log_var'] +
                     ((post['mean'] - prior['mean']).pow(2) + post_var) /
                     (2 * prior_var + small_num) - 0.5)

    return kld

# -------------------------------------------------------------------------------------------
#  Intra-task complexity for deterministic posterior (point-wise)
# -------------------------------------------------------------------------------------------
def get_weights_complexity_term(complexity_type, prior_means_model, prior_log_vars_model, task_post_model, n_samples_task):

    neg_log_pdf = calc_neg_log_pdf(prior_means_model, prior_log_vars_model, task_post_model)

    if complexity_type == 'Variational_Bayes':
        complex_term = (1 / n_samples_task) * neg_log_pdf

    elif complexity_type == 'PAC_Bayes':
        complex_term = torch.sqrt((1 / n_samples_task) * (neg_log_pdf + 10))

    else:

        raise ValueError('Invalid complexity_type')

    return complex_term



def calc_neg_log_pdf(prior_means_model, prior_log_vars_model, task_post_model):
    # Calculate the log-probability of the posterior weights vector given a factorized Gaussian distribution
    # (the prior) with a given mean and log-variance vectors

    param_names_list = [param_name for param_name, param in prior_means_model.named_parameters()]

    small_num = 1e-20  # add small positive number to avoid division by zero due to numerical errors
    neg_log_pdf = 0

    # Since the distribution is factorized, we sum the log-prob over all elements:
    for i_param, param_name in enumerate(param_names_list):
        # Get the corresponding prior mean parameters of the current weight group:
        w_P_mu = get_param_from_model(prior_means_model, param_name)
        # Get the corresponding prior log-var parameters of the current weight group:
        w_P_log_var = get_param_from_model(prior_log_vars_model, param_name)
        # Get the corresponding posterior weight parameters of the current weight group:
        w_post = get_param_from_model(task_post_model, param_name)

        sigma_sqr_prior = torch.exp(w_P_log_var)

        neg_log_pdf_curr = 0.5 * torch.sum(w_P_log_var + np.log(2*np.pi) +
                                           (w_post - w_P_mu).pow(2) / (2*sigma_sqr_prior + small_num))

        # Sum the contribution to the total log-probability
        neg_log_pdf += neg_log_pdf_curr

    return neg_log_pdf


