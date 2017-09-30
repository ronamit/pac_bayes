from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class StochasticLinear(nn.Module):

    def __init__(self, in_dim, out_dim, prm):
        super(self.__class__, self).__init__()

        rand_init_std = prm.rand_init_std
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w_mu = nn.Parameter(torch.randn(
            out_dim, in_dim) * rand_init_std)
        self.w_log_sigma = nn.Parameter(
            torch.randn(out_dim, in_dim) * rand_init_std)
        self.b_mu = nn.Parameter(torch.randn(out_dim) * rand_init_std)
        self.b_log_sigma = nn.Parameter(torch.randn(out_dim) * rand_init_std)



    def __str__(self):
        return 'StochasticLinear({0} -> {1})'.format(self.in_dim, self.out_dim)

    def forward(self, x, eps_std):

        # Layer computations (based on "Variational Dropout and the Local
        # Reparameterization Trick", Kingma et.al 2015)

        out_mean = F.linear(x, self.w_mu, bias=self.b_mu)

        if eps_std == 0.0:
            layer_out = out_mean
        else:
            w_sigma_sqr = torch.exp(2 * self.w_log_sigma)
            b_sigma_sqr = torch.exp(2 * self.b_log_sigma)

            out_var = F.linear(x.pow(2), w_sigma_sqr, bias=b_sigma_sqr)

            # Draw Gaussian random noise, N(0, eps_std) in the size of the
            # layer output:
            noise = out_mean.data.new(out_mean.size()).normal_(0, eps_std)
            ksi = Variable(noise, requires_grad=False)

            layer_out = out_mean + eps_std * ksi * torch.sqrt(out_var)

        return layer_out
