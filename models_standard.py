
from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

import data_gen


# Note: the net return scores (not normalized probabilities)

def get_model(model_type, prm):

    info = data_gen.get_info(prm)
    color_channels = info['color_channels']
    im_size =info['im_size']
    n_classes = info['n_classes']
    input_size = info['input_size']

    # -------------------------------------------------------------------------------------------
    #  ConvNet
    # -------------------------------------------------------------------------------------------
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            n_filt1 = 10
            n_filt2 = 20
            n_hidden_fc1 = 50
            self.conv1 = nn.Conv2d(color_channels, n_filt1, kernel_size=5)
            self.conv2 = nn.Conv2d(n_filt1, n_filt2, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            n_conv_size = self._get_conv_output()
            self.fc1 = nn.Linear(n_conv_size, n_hidden_fc1)
            self.fc_out = nn.Linear(n_hidden_fc1, n_classes)

        # generate dummy input sample and forward to get shape after conv layers
        def _get_conv_output(self):
            batch_size = 1
            input_shape = (color_channels, im_size, im_size)
            input = Variable(torch.rand(batch_size, *input_shape))
            output_feat = self._forward_features(input)
            n_conv_size = output_feat.data.view(batch_size, -1).size(1)
            return n_conv_size

        def _forward_features(self, x):
            x = F.elu(F.max_pool2d(self.conv1(x), 2))
            x = F.elu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            return x

        def forward(self, x):
            x = self._forward_features(x)
            x = x.view(x.size(0), -1)
            x = F.elu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc_out(x)
            return x

    # -------------------------------------------------------------------------------------------
    #  Fully-Connected Net
    # -------------------------------------------------------------------------------------------
    class FcNet(nn.Module):
        def __init__(self):
            super(FcNet, self).__init__()
            self.model_type = model_type

            n_hidden1 = 800
            n_hidden2 = 800
            self.fc1 = nn.Linear(input_size, n_hidden1)
            self.fc2 = nn.Linear(n_hidden1, n_hidden2)
            self.fc_out = nn.Linear(n_hidden2, n_classes)

        def forward(self, x):
            x = x.view(-1, input_size)  # flatten image
            x = F.elu(self.fc1(x))
            x = F.elu(self.fc2(x))
            x = self.fc_out(x)
            return x

    class FcNet3(nn.Module):
        def __init__(self):
            super(FcNet3, self).__init__()
            self.model_type = model_type

            n_hidden1 = 400
            n_hidden2 = 400
            n_hidden3 = 400
            self.fc1 = nn.Linear(input_size, n_hidden1)
            self.fc2 = nn.Linear(n_hidden1, n_hidden2)
            self.fc3 = nn.Linear(n_hidden2, n_hidden3)
            self.fc_out = nn.Linear(n_hidden3, n_classes)

        def forward(self, x):
            x = x.view(-1, input_size)  # flatten image
            x = F.elu(self.fc1(x))
            x = F.elu(self.fc2(x))
            x = F.elu(self.fc3(x))
            x = self.fc_out(x)
            return x

    models_dict = {'FcNet':FcNet(), 'ConvNet':ConvNet(), 'FcNet3':FcNet3()}
    model = models_dict[model_type]


    if prm.cuda:
        model.cuda()

    return model