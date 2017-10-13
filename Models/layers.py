from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Utils.common import randn_gpu


# -------------------------------------------------------------------------------------------
#  Auxilary functions
# -------------------------------------------------------------------------------------------
def make_pair(x):
    if isinstance(x, int):
        return (x, x)
    else:
        return x

def get_randn_param(shape, mean, std):
    if isinstance(shape, int):
        shape = (shape,)
    return nn.Parameter(torch.FloatTensor(*shape).normal_(mean, std))


def init_stochastic_layer(layer, weights_size, bias_size, prm):
    
    inits = prm.bayes_inits
    mu_bias = inits['Bayes-Mu']['bias']
    mu_std = inits['Bayes-Mu']['std']
    log_var_bias = inits['Bayes-log-var']['bias']
    log_var_std = inits['Bayes-log-var']['std']
    
    layer.w_mu = get_randn_param(weights_size, mu_bias, mu_std)
    layer.w_log_var = get_randn_param(weights_size, log_var_bias, log_var_std)
    layer.b_mu = get_randn_param(bias_size, mu_bias, mu_std)
    layer.b_log_var = get_randn_param(bias_size, log_var_bias, log_var_std)

    layer.w = {'mean': layer.w_mu, 'log_var': layer.w_log_var}
    layer.b = {'mean': layer.b_mu, 'log_var': layer.b_log_var}


# -------------------------------------------------------------------------------------------
#  Stochastic linear layer
# -------------------------------------------------------------------------------------------
class StochasticLinear(nn.Module):

    def __init__(self, in_dim, out_dim, prm):
        super(self.__class__, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        weights_size = (out_dim, in_dim)
        bias_size = out_dim
        init_stochastic_layer(self, weights_size, bias_size, prm)

    def __str__(self):
        return 'StochasticLinear({0} -> {1})'.format(self.in_dim, self.out_dim)

    def forward(self, x, eps_std=1.0):

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

# -------------------------------------------------------------------------------------------
#  Stochastic conv2d layer
# -------------------------------------------------------------------------------------------

class StochasticConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, prm):
        super(self.__class__, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = make_pair(kernel_size)
        self.kernel_size = kernel_size

        weights_size = (out_channels, in_channels, *kernel_size)
        bias_size = (out_channels)
        init_stochastic_layer(self, weights_size, bias_size, prm)


    def __str__(self):
        return 'StochasticConv2d({} -> {}, kernel_size={})'.format(self.in_channels, self.out_channels, self.kernel_size)

    def forward(self, x, eps_std=1.0):

        # Layer computations (based on "Variational Dropout and the Local
        # Reparameterization Trick", Kingma et.al 2015)

        out_mean = F.conv2d(x, self.w['mean'], bias=self.b['mean'])

        if eps_std == 0.0:
            layer_out = out_mean
        else:
            w_var = torch.exp(self.w_log_var)
            b_var = torch.exp(self.b_log_var)
            out_var = F.conv2d(x.pow(2), w_var, bias=b_var)

            # Draw Gaussian random noise, N(0, eps_std) in the size of the
            # layer output:
            noise = out_mean.data.new(out_mean.size()).normal_(0, eps_std)
            # noise = randn_gpu(size=out_mean.size(), mean=0, std=eps_std)
            noise = Variable(noise, requires_grad=False)

            layer_out = out_mean + noise * torch.sqrt(out_var)

        return layer_out


