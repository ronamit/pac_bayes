
import math

import torch

from Models.stochastic_layers import StochasticLayer


# -----------------------------------------------------------------------------------------------------------#


# -----------------------------------------------------------------------------------------------------------#

def get_task_complexity(prm, prior_model, post_model, n_samples, avg_empiric_loss, dvrg=None,
                        noised_prior=False):  # corrected
    #  Intra-task complexity for posterior distribution

    complexity_type = prm.complexity_type
    delta = prm.delta  # maximal probability that the bound does not hold

    if dvrg is None:
        # calculate divergence between posterior and sampled prior
        dvrg = get_net_densities_divergence(prior_model, post_model, prm, noised_prior)

    if complexity_type == 'NoComplexity':
        # set as zero
        complex_term = torch.tensor(0., device=prm.device)

    elif prm.complexity_type in {'McAllester', 'Classic_PB'}:
        # According to 'Simplified PAC-Bayesian Margin Bounds', McAllester 2003
        complex_term = torch.sqrt((dvrg + math.log(2 * n_samples / delta)) / (2 * (n_samples - 1)))  # corrected

    elif prm.complexity_type in {'New_PB'}:
        # According to 'Simplified PAC-Bayesian Margin Bounds', McAllester 2003
        kl = dvrg
        delta_ub = delta / 2
        classic_pb = (kl + math.log(2 * n_samples / delta_ub)) / (2 * (n_samples - 1))
        pinsker_pb = torch.sqrt(0.5 * kl) + (math.log(2 * n_samples / delta_ub)) / (2 * (n_samples - 1))
        bh_pb = torch.sqrt(1 - torch.exp(-kl)) + (math.log(2 * n_samples / delta_ub)) / (2 * (n_samples - 1))
        sqrt_arg = torch.minimum(classic_pb, pinsker_pb)
        sqrt_arg = torch.minimum(sqrt_arg, bh_pb)
        complex_term = torch.sqrt(sqrt_arg)

    elif prm.complexity_type == 'Seeger':
        # According to 'Simplified PAC-Bayesian Margin Bounds', McAllester 2003
        seeger_eps = (dvrg + math.log(2 * math.sqrt(n_samples) / delta)) / n_samples  # corrected

        sqrt_arg = 2 * seeger_eps * avg_empiric_loss
        # sqrt_arg = F.relu(sqrt_arg)  # prevent negative values due to numerical errors
        complex_term = 2 * seeger_eps + torch.sqrt(sqrt_arg)

    elif prm.complexity_type == 'Catoni':
        # See "From PAC-Bayes Bounds to KL Regularization" Germain 2009
        # & Olivier Catoni. PAC-Bayesian surpevised classification: the thermodynamics of statistical learning
        complex_term = avg_empiric_loss + (2 / n_samples) * (dvrg + math.log(1 / delta))  # corrected

    else:
        raise ValueError('Invalid complexity_type')

    return complex_term


# -------------------------------------------------------------------------------------------


def get_net_densities_divergence(prior_model, post_model, prm, noised_prior=False):
    prior_layers_list = [layer for layer in prior_model.children() if isinstance(layer, StochasticLayer)]
    post_layers_list = [layer for layer in post_model.children() if isinstance(layer, StochasticLayer)]

    total_dvrg = torch.tensor(0., device=prm.device)
    for i_layer, prior_layer in enumerate(prior_layers_list):
        post_layer = post_layers_list[i_layer]
        if hasattr(prior_layer, 'w'):
            total_dvrg += get_dvrg_element(post_layer.w, prior_layer.w, prm, noised_prior)
        if hasattr(prior_layer, 'b'):
            total_dvrg += get_dvrg_element(post_layer.b, prior_layer.b, prm, noised_prior)

    if prm.divergence_type == 'W_NoSqr':
        total_dvrg = torch.sqrt(total_dvrg)

    return total_dvrg


# -------------------------------------------------------------------------------------------

def get_dvrg_element(post, prior, prm, noised_prior=False):
    """KL divergence D_{KL}[post(x)||prior(x)] for a fully factorized Gaussian"""

    if noised_prior and prm.kappa_post > 0:
        prior_log_var = add_noise(prior['log_var'], prm.kappa_post)
        prior_mean = add_noise(prior['mean'], prm.kappa_post)
    else:
        prior_log_var = prior['log_var']
        prior_mean = prior['mean']

    post_var = torch.exp(post['log_var'])
    prior_var = torch.exp(prior_log_var)
    post_std = torch.exp(0.5 * post['log_var'])
    prior_std = torch.exp(0.5 * prior_log_var)

    if prm.divergence_type in ['W_Sqr', 'W_NoSqr']:
        # Wasserstein norm with p=2
        # according to DOWSON & LANDAU 1982
        div_elem = torch.sum((post['mean'] - prior_mean).pow(2) + (post_std - prior_std).pow(2))

    elif prm.divergence_type == 'KL':
        numerator = (post['mean'] - prior_mean).pow(2) + post_var
        denominator = prior_var
        div_elem = 0.5 * torch.sum(prior_log_var - post['log_var'] + numerator / denominator - 1)
    else:
        raise ValueError('Invalid prm.divergence_type')

    # note: don't add small number to denominator, since we need to have zero KL when post==prior.

    assert div_elem >= 0
    return div_elem


# -------------------------------------------------------------------------------------------

def add_noise(param, std):
    param += torch.randn_like(param) * std


# -------------------------------------------------------------------------------------------

def add_noise_to_model(model, std):
    layers_list = [layer for layer in model.children() if isinstance(layer, StochasticLayer)]

    for i_layer, layer in enumerate(layers_list):
        if hasattr(layer, 'w'):
            add_noise(layer.w['log_var'], std)
            add_noise(layer.w['mean'], std)
        if hasattr(layer, 'b'):
            add_noise(layer.b['log_var'], std)
            add_noise(layer.b['mean'], std)
