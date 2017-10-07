from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class StochasticLinear(nn.Module):

    def __init__(self, in_dim, out_dim, prm):
        super(self.__class__, self).__init__()

        log_var_init_std = prm.log_var_init_std
        mu_init_std = prm.mu_init_std
        log_var_init_bias= prm.log_var_init_bias
        mu_init_bias = prm.mu_init_bias
        self.in_dim = in_dim
        self.out_dim = out_dim

        # self.w_mean = nn.Parameter(randn_gpu(out_dim, in_dim) * rand_init_std)
        # self.w_log_var = nn.Parameter(randn_gpu(out_dim, in_dim) * rand_init_std)
        #
        # self.b_mean = nn.Parameter(randn_gpu(out_dim) * rand_init_std)
        # self.b_log_var = nn.Parameter(randn_gpu(out_dim) * rand_init_std)

        self.w_mean = nn.Parameter(torch.randn(out_dim, in_dim) * mu_init_std + mu_init_std)
        self.w_log_var = nn.Parameter(torch.randn(out_dim, in_dim) * log_var_init_std + log_var_init_bias)

        self.b_mean = nn.Parameter(torch.randn(out_dim) * mu_init_std + mu_init_std)
        self.b_log_var = nn.Parameter(torch.randn(out_dim) * log_var_init_std + log_var_init_bias)


        self.w = {'mean': self.w_mean, 'log_var': self.w_log_var}
        self.b = {'mean': self.b_mean, 'log_var': self.b_log_var}

        # self.weights_groups_list = [self.w, self.b]

    def __str__(self):
        return 'StochasticLinear({0} -> {1})'.format(self.in_dim, self.out_dim)

    def forward(self, x, eps_std):

        # Layer computations (based on "Variational Dropout and the Local
        # Reparameterization Trick", Kingma et.al 2015)

        out_mean = F.linear(x, self.w['mean'], bias=self.b['mean'])

        if eps_std == 0.0:
            layer_out = out_mean
        else:
            w_var = torch.exp(self.w_log_var)
            b_var = torch.exp(self.b_log_var)
            out_var = F.linear(x.pow(2), w_var, bias=b_var)

            # Draw Gaussian random noise, N(0, eps_std) in the size of the
            # layer output:
            noise = out_mean.data.new(out_mean.size()).normal_(0, eps_std)
            # noise = randn_gpu(size=out_mean.size(), mean=0, std=eps_std)
            noise = Variable(noise, requires_grad=False)

            layer_out = out_mean + noise * torch.sqrt(out_var)

        return layer_out
