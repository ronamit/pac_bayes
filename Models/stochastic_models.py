#
# the code is inspired by: https://github.com/katerakelly/pytorch-maml

import numpy as np
import random
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Utils import data_gen

from Models.stochastic_layers import StochasticLinear, StochasticConv2d, StochasticLayer

# -------------------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------------------
def get_model(prm):
    model_name = prm.model_name
    # Get task info:
    info = data_gen.get_info(prm)
    input_shape = info['input_shape']
    color_channels = input_shape[0]
    n_classes = info['n_classes']
    input_size = input_shape[0] * input_shape[1] * input_shape[2]

    if model_name == 'FcNet3':
        model = FcNet3(prm, input_size=input_size, n_classes=n_classes)
    elif model_name == 'ConvNet3':
        model = ConvNet3(prm, input_shape=input_shape, n_classes=n_classes)
    else:
        raise ValueError('Invalid model_name')

    return model

# -------------------------------------------------------------------------------------------
# Auxiliary functions
# -------------------------------------------------------------------------------------------

# generate dummy input sample and forward to get shape after conv layers
def get_size_of_conv_output(input_shape, conv_func):
    batch_size = 1
    input = Variable(torch.rand(batch_size, *input_shape))
    output_feat = conv_func(input)
    conv_out_size = output_feat.data.view(batch_size, -1).size(1)
    return conv_out_size


# -------------------------------------------------------------------------------------------
#  Base class for all stochastic models
# -------------------------------------------------------------------------------------------
class base_stochastic_model(nn.Module):
    def __init__(self, prm, *args, **kwargs):
        super(base_stochastic_model, self).__init__()
        self.model_type = 'Stochastic'

        #  Stochastic Layers Wrappers
        self.s_Linear = lambda in_dim, out_dim: StochasticLinear(in_dim, out_dim, prm)
        self.s_Conv2d = lambda in_channels, out_channels, kernel_size, use_bias=False, stride=1, padding=0, dilation=1:\
            StochasticConv2d(in_channels, out_channels, kernel_size, prm, use_bias, stride, padding, dilation)


    def set_eps_std(self, eps_std=1.0):
        ''' Set the  epsilon STD for re-parametrization trick.
            Usually, eps_std sould be 1,
            value of 0 means - use only the max-posterior'''
        old_eps_std = None
        for m in self.modules():
            if isinstance(m, StochasticLayer):
                old_eps_std = m.set_eps_std(eps_std)
        return old_eps_std

    def net_forward(self, x):
        return self.forward(x)  # forward is defined in derived classes

    # def _init_weights(self):
    #     ''' Set weights to Gaussian, biases to zero '''
    #
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             # Similar to PyTorch' default initializer
    #             n = m.weight.size(1)
    #             stdv = 1. / math.sqrt(n)
    #             m.weight.data.uniform_(-stdv, stdv)
    #             # m.weight.data.normal_(0, 0.01)
    #             if m.bias is not None:
    #                 m.bias.data.uniform_(1-stdv, 1+stdv)
    #                 # m.bias.data = torch.ones(m.bias.data.size())

    # def copy_weights(self, net):
    #     ''' Set this module's weights to be the same as those of 'net' '''
    #     # TODO: breaks if nets are not identical
    #     # TODO: won't copy buffers, e.g. for batch norm
    #     for m_from, m_to in zip(net.modules(), self.modules()):
    #         if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
    #             m_to.weight.data = m_from.weight.data.clone()
    #             if m_to.bias is not None:
    #                 m_to.bias.data = m_from.bias.data.clone()



# -------------------------------------------------------------------------------------------
#  3-hidden-layer Fully-Connected Net
# -------------------------------------------------------------------------------------------
class FcNet3(base_stochastic_model):
    def __init__(self, prm, input_size, n_classes):
        super(FcNet3, self).__init__(prm)
        self.model_name = 'FcNet3'
        self.input_size = input_size
        n_hidden1 = 400
        n_hidden2 = 400
        n_hidden3 = 400
        self.net = nn.Sequential(OrderedDict([
                ('fc1',  self.s_Linear(input_size, n_hidden1)),
                ('a1',  nn.ELU(inplace=True)),
                ('fc2',  self.s_Linear(n_hidden1, n_hidden2)),
                ('a2', nn.ELU(inplace=True)),
                ('fc3',  self.s_Linear(n_hidden2, n_hidden3)),
                ('a3', nn.ELU(inplace=True)),
                ('fc_out', self.s_Linear(n_hidden3, n_classes)),
                ]))
        # Initialize weights
        # self._init_weights()
        self.cuda()  # always use GPU

    def forward(self, x):
        ''' Define what happens to data in the net '''
        x = x.view(-1, self.input_size)  # flatten image
        x = self.net(x)
        return x


# -------------------------------------------------------------------------------------------
#  3-hidden-layer ConvNet
# -------------------------------------------------------------------------------- -----------
class ConvNet3(base_stochastic_model):
    def __init__(self, prm, input_shape, n_classes):
        super(ConvNet3, self).__init__(prm)
        self.model_name = 'ConvNet3'
        n_in_channels = input_shape[0]
        n_filt1 = 10
        n_filt2 = 20
        n_hidden_fc1 = 50
        self.conv_layers = nn.Sequential(OrderedDict([
                ('conv1',  self.s_Conv2d(n_in_channels, n_filt1, kernel_size=5)),
                ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('a1',  nn.ELU(inplace=True)),
                ('conv2', self.s_Conv2d(n_filt1, n_filt2, kernel_size=5)),
                ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('a2', nn.ELU(inplace=True)),
                 ]))
        conv_out_size = get_size_of_conv_output(input_shape, self._forward_conv_layers)
        self.add_module('fc1', self.s_Linear(conv_out_size, n_hidden_fc1))
        self.add_module('a3', nn.ELU(inplace=True)),
        self.add_module('fc_out', self.s_Linear(n_hidden_fc1, n_classes))

        # Initialize weights
        # self._init_weights()
        self.cuda()  # always use GPU

    def _forward_conv_layers(self, x):
        x = self.conv_layers(x)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc_out(x)
        return x
