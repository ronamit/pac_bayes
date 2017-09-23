from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


import data_gen
from layers import StochasticLinear

# Note: the net return scores (not normalized probabilities)


def get_model(model_type, prm):

    info = data_gen.get_info(prm)
    color_channels = info['color_channels']
    im_size =info['im_size']
    n_classes = info['n_classes']
    input_size = info['input_size']

    # -------------------------------------------------------------------------------------------
    #  Fully-Connected Net - Stochastic
    # -------------------------------------------------------------------------------------------
    class BayesNN(nn.Module):
        def __init__(self):
            super(self.__class__, self).__init__()
            n_hidden1 = 800
            n_hidden2 = 800
            self.fc1 = StochasticLinear(input_size, n_hidden1)
            self.fc2 = StochasticLinear(n_hidden1, n_hidden2)
            self.fc_out = StochasticLinear(n_hidden2, n_classes)

        def forward(self, x, eps_std):
            x = x.view(-1, input_size)  # flatten image
            x = F.elu(self.fc1(x, eps_std))
            x = F.elu(self.fc2(x, eps_std))
            x = self.fc_out(x, eps_std)
            return x


    class BigBayesNN(nn.Module):
        def __init__(self):
            super(self.__class__, self).__init__()
            n_hidden1 = 800
            n_hidden2 = 800
            n_hidden3 = 800
            self.fc1 = StochasticLinear(input_size, n_hidden1)
            self.fc2 = StochasticLinear(n_hidden1, n_hidden2)
            self.fc3 = StochasticLinear(n_hidden2, n_hidden3)
            self.fc_out = StochasticLinear(n_hidden3, n_classes)

        def forward(self, x, eps_std):
            x = x.view(-1, input_size)  # flatten image
            x = F.elu(self.fc1(x, eps_std))
            x = F.elu(self.fc2(x, eps_std))
            x = F.elu(self.fc3(x, eps_std))
            x = self.fc_out(x, eps_std)
            return x


    # -------------------------------------------------------------------------------------------
    #  Return net
    # -------------------------------------------------------------------------------------------

    if model_type == 'BayesNN':
        model = BayesNN()
    elif model_type == 'BigBayesNN':
        model = BigBayesNN()
    else:
        raise ValueError('Invalid model_type')

    if prm.cuda:
        model.cuda()

    return model