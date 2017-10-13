
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Utils import data_gen
from Models.layers import StochasticLinear, StochasticConv2d


# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

def get_model(prm, model_type, init_override=None):

    model_name = prm.model_name

    # Get task info:
    info = data_gen.get_info(prm)
    color_channels = info['color_channels']
    im_size = info['im_size']
    n_classes = info['n_classes']
    input_size = info['input_size']

    def linear_layer(in_dim, out_dim):
        if model_type == 'Standard':
            return nn.Linear(in_dim, out_dim)
        elif model_type == 'Stochastic':
            return StochasticLinear(in_dim, out_dim, prm)

    def conv2d_layer(in_channels, out_channels, kernel_size):
        if model_type == 'Standard':
            return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        elif model_type == 'Stochastic':
            return StochasticConv2d(in_channels, out_channels, kernel_size, prm)

    # -------------------------------------------------------------------------------------------
    #  2-hidden-layer Fully-Connected Net
    # -------------------------------------------------------------------------------------------
    class FcNet2(nn.Module):
        def __init__(self):
            super(FcNet2, self).__init__()
            self.model_type = model_type
            self.model_name = model_name

            n_hidden1 = 800
            n_hidden2 = 800
            self.fc1 = linear_layer(input_size, n_hidden1)
            self.fc2 = linear_layer(n_hidden1, n_hidden2)
            self.fc_out = linear_layer(n_hidden2, n_classes)


        def forward(self, x, *a):
            x = x.view(-1, input_size)  # flatten image
            x = F.elu(self.fc1(x, *a))
            x = F.elu(self.fc2(x, *a))
            x = self.fc_out(x, *a)
            return x

    # -------------------------------------------------------------------------------------------
    #  3-hidden-layer Fully-Connected Net
    # -------------------------------------------------------------------------------------------
    class FcNet3(nn.Module):
        def __init__(self):
            super(FcNet3, self).__init__()
            self.model_type = model_type
            self.model_name = model_name

            n_hidden1 = 400
            n_hidden2 = 400
            n_hidden3 = 400
            self.fc1 = linear_layer(input_size, n_hidden1)
            self.fc2 = linear_layer(n_hidden1, n_hidden2)
            self.fc3 = linear_layer(n_hidden2, n_hidden3)
            self.fc_out = linear_layer(n_hidden3, n_classes)

        def forward(self, x, *a):
            x = x.view(-1, input_size)  # flatten image
            x = F.elu(self.fc1(x, *a))
            x = F.elu(self.fc2(x, *a))
            x = F.elu(self.fc3(x, *a))
            x = self.fc_out(x)
            return x

            # -------------------------------------------------------------------------------------------
            #  ConvNet with dropout
            # -------------------------------------------------------------------------------------------

    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.model_type = model_type
            self.model_name = model_name
            n_filt1 = 10
            n_filt2 = 20
            n_hidden_fc1 = 50
            self.conv1 = conv2d_layer(color_channels, n_filt1, kernel_size=5)
            self.conv2 = conv2d_layer(n_filt1, n_filt2, kernel_size=5)
            n_conv_size = self._get_conv_output()
            self.fc1 = linear_layer(n_conv_size, n_hidden_fc1)
            self.fc_out = linear_layer(n_hidden_fc1, n_classes)

        # generate dummy input sample and forward to get shape after conv layers
        def _get_conv_output(self):
            batch_size = 1
            input_shape = (color_channels, im_size, im_size)
            input = Variable(torch.rand(batch_size, *input_shape))
            output_feat = self._forward_features(input)
            n_conv_size = output_feat.data.view(batch_size, -1).size(1)
            return n_conv_size

        def _forward_features(self, x, *a):
            x = F.elu(F.max_pool2d(self.conv1(x, *a), 2))
            x = F.elu(F.max_pool2d(self.conv2(x, *a), 2))
            return x

        def forward(self, x, *a):
            x = self._forward_features(x)
            x = x.view(x.size(0), -1)
            x = F.elu(self.fc1(x, *a))
            x = F.dropout(x, training=self.training)
            x = self.fc_out(x, *a)
            return x

    # -------------------------------------------------------------------------------------------
    #  ConvNet with dropout
    # -------------------------------------------------------------------------------------------
    class ConvNet_Dropout(nn.Module):
        def __init__(self):
            super(ConvNet_Dropout, self).__init__()
            self.model_type = model_type
            self.model_name = model_name
            n_filt1 = 10
            n_filt2 = 20
            n_hidden_fc1 = 50
            self.conv1 = conv2d_layer(color_channels, n_filt1, kernel_size=5)
            self.conv2 = conv2d_layer(n_filt1, n_filt2, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            n_conv_size = self._get_conv_output()
            self.fc1 = linear_layer(n_conv_size, n_hidden_fc1)
            self.fc_out = linear_layer(n_hidden_fc1, n_classes)

        # generate dummy input sample and forward to get shape after conv layers
        def _get_conv_output(self):
            batch_size = 1
            input_shape = (color_channels, im_size, im_size)
            input = Variable(torch.rand(batch_size, *input_shape))
            output_feat = self._forward_features(input)
            n_conv_size = output_feat.data.view(batch_size, -1).size(1)
            return n_conv_size

        def _forward_features(self, x, *a):
            x = F.elu(F.max_pool2d(self.conv1(x, *a), 2))
            x = F.elu(F.max_pool2d(self.conv2_drop(self.conv2(x, *a)), 2))
            return x

        def forward(self, x, *a):
            x = self._forward_features(x)
            x = x.view(x.size(0), -1)
            x = F.elu(self.fc1(x, *a))
            x = F.dropout(x, training=self.training)
            x = self.fc_out(x, *a)
            return x

    # -------------------------------------------------------------------------------------------
    #  Return selected model:
    # -------------------------------------------------------------------------------------------

    if model_name == 'FcNet2':
        model = FcNet2()
    elif model_name == 'FcNet3':
        model = FcNet3()
    elif model_name == 'ConvNet':
        model = ConvNet()
    elif  model_name == 'ConvNet_Dropout':
        model = ConvNet_Dropout()
    else:
        raise ValueError('Invalid model_name')

    if init_override:
        print('Initializing model with override values: {}'.format(init_override))
        for param in model.parameters():
            param.data.normal_(mean=init_override['mean'], std=init_override['std'])

    if prm.cuda:
        model.cuda()

    return model