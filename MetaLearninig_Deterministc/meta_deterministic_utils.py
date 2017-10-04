
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

#  -----------
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


