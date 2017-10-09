
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import math
import torch.nn.functional as F
from Utils.common import get_param_from_model


#  -----------
# -------------------------------------------------------------------------------------------
#  Intra-task complexity for deterministic posterior (point-wise)
# -------------------------------------------------------------------------------------------
def get_weights_complexity_term(complexity_type, prior_means_model, prior_log_vars_model, task_post_model, n_samples):

    neg_log_pdf = calc_neg_log_pdf(prior_means_model, prior_log_vars_model, task_post_model)

    if complexity_type == 'Variational_Bayes':
        complex_term = (1 / n_samples) * neg_log_pdf

    elif complexity_type == 'PAC_Bayes_McAllaster':
        complex_term = torch.sqrt((1 / n_samples) * F.relu(neg_log_pdf + 1e10)) - np.sqrt((1 / n_samples) * (1e10))
        # I subtracted a const. to keep values around zero

    elif complexity_type == 'PAC_Bayes_Pentina':
        complex_term = math.sqrt(1 / n_samples) * neg_log_pdf

    else:

        raise ValueError('Invalid complexity_type')

    return complex_term



def calc_neg_log_pdf(prior_means_model, prior_log_vars_model, task_post_model):
    # Calculate the log-probability of the posterior weights vector given a factorized Gaussian distribution
    # (the prior) with a given mean and log-variance vectors

    param_names_list = [param_name for param_name, param in prior_means_model.named_parameters()]


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
                                           (w_post - w_P_mu).pow(2) / (2*sigma_sqr_prior ))

        # Sum the contribution to the total log-probability
        neg_log_pdf += neg_log_pdf_curr

    return neg_log_pdf


