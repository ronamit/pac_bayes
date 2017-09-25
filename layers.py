from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



class StochasticLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(self.__class__, self).__init__()
        random_init_std = 0.1

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w_mu = nn.Parameter(torch.randn(in_dim, out_dim) * random_init_std)
        self.w_log_sigma = nn.Parameter(torch.randn(in_dim, out_dim) * random_init_std)
        self.b_mu = nn.Parameter(torch.randn(out_dim) * random_init_std)
        self.b_log_sigma = nn.Parameter(torch.randn(out_dim) *random_init_std)

        # Weights initial values:
        # random_init_std = 0.1
        # nn.init.normal(self.w_mu, 0, random_init_std)
        # nn.init.normal(self.w_log_sigma, 0, random_init_std)
        # nn.init.normal(self.b_mu, 0, random_init_std)
        # nn.init.normal(self.b_log_sigma, 0, random_init_std)

    def __str__(self):
        return 'StochasticLinear({0} -> {1})'.format(self.in_dim, self.out_dim)

    def forward(self, x, eps_std):

        # Layer computations (based on "Variational Dropout and the Local Reparameterization Trick", Kingma et.al 2015)

        out_mean = torch.mm(x, self.w_mu) + self.b_mu

        if eps_std == 0.0:
            layer_out = out_mean
        else:
            w_sigma_sqr = torch.exp(2 * self.w_log_sigma)
            b_sigma_sqr = torch.exp(2 * self.b_log_sigma)

            out_var = torch.mm(x.pow(2), w_sigma_sqr) + b_sigma_sqr

            # Draw Gaussian random noise, N(0, eps_std) in the size of the layer output:
            ksi = Variable(torch.randn(out_mean.size()).cuda(), requires_grad=False)

            layer_out = out_mean + eps_std * ksi * torch.sqrt(out_var)

        return layer_out

