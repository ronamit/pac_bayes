
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Utils import data_gen
from Models.layers import StochasticLinear, StochasticConv2d, general_model
import torchvision.models
from Models.densenet import get_densenet_model_class
from Models.densenetBayes import get_bayes_densenet_model_class

# -------------------------------------------------------------------------------------------
# Auxiliary functions
# -------------------------------------------------------------------------------------------

# generate dummy input sample and forward to get shape after conv layers
def get_size_of_conv_output(input_shape, conv_func):
    batch_size = 1
    input = Variable(torch.rand(batch_size, *input_shape))
    output_feat = conv_func(input)
    conv_feat_size = output_feat.data.view(batch_size, -1).size(1)
    return conv_feat_size

# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------

def get_model(prm, model_type, init_override=None):

    model_name = prm.model_name

    # Get task info:
    info = data_gen.get_info(prm)
    input_shape = info['input_shape']
    color_channels = input_shape[0]
    n_classes = info['n_classes']
    input_size = input_shape[0] * input_shape[1] * input_shape[2]

    def linear_layer(in_dim, out_dim):
        if model_type == 'Standard':
            return nn.Linear(in_dim, out_dim)
        elif model_type == 'Stochastic':
            return StochasticLinear(in_dim, out_dim, prm)

    def conv2d_layer(in_channels, out_channels, kernel_size, use_bias=False, stride=1, padding=0, dilation=1):
        if model_type == 'Standard':
            return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        elif model_type == 'Stochastic':
            return StochasticConv2d(in_channels, out_channels, kernel_size, prm, use_bias, stride, padding, dilation)

    # -------------------------------------------------------------------------------------------
    #  2-hidden-layer Fully-Connected Net
    # -------------------------------------------------------------------------------------------
    class FcNet2(general_model):
        def __init__(self):
            super(FcNet2, self).__init__()
            self.model_type = model_type
            self.model_name = model_name

            n_hidden1 = 800
            n_hidden2 = 800
            self.fc1 = linear_layer(input_size, n_hidden1)
            self.fc2 = linear_layer(n_hidden1, n_hidden2)
            self.fc_out = linear_layer(n_hidden2, n_classes)


        def forward(self, x):
            x = x.view(-1, input_size)  # flatten image
            x = F.elu(self.fc1(x))
            x = F.elu(self.fc2(x))
            x = self.fc_out(x)
            return x

    # -------------------------------------------------------------------------------------------
    #  3-hidden-layer Fully-Connected Net
    # -------------------------------------------------------------------------------------------
    class FcNet3(general_model):
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

        def forward(self, x):
            x = x.view(-1, input_size)  # flatten image
            x = F.elu(self.fc1(x))
            x = F.elu(self.fc2(x))
            x = F.elu(self.fc3(x))
            x = self.fc_out(x)
            return x

    # -------------------------------------------------------------------------------------------
    #  ConvNet
    # -------------------------------------------------------------------------------- -----------

    class ConvNet(general_model):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.model_type = model_type
            self.model_name = model_name
            n_filt1 = 10
            n_filt2 = 20
            n_hidden_fc1 = 50
            self.conv1 = conv2d_layer(color_channels, n_filt1, kernel_size=5)
            self.conv2 = conv2d_layer(n_filt1, n_filt2, kernel_size=5)
            conv_feat_size =  get_size_of_conv_output(input_shape, self._forward_features)
            self.fc1 = linear_layer(conv_feat_size, n_hidden_fc1)
            self.fc_out = linear_layer(n_hidden_fc1, n_classes)

        def _forward_features(self, x):
            x = F.elu(F.max_pool2d(self.conv1(x), 2))
            x = F.elu(F.max_pool2d(self.conv2(x), 2))
            return x

        def forward(self, x):
            x = self._forward_features(x)
            x = x.view(x.size(0), -1)
            x = F.elu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc_out(x)
            return x

    # -------------------------------------------------------------------------------------------
    #  ConvNet with dropout
    # -------------------------------------------------------------------------------------------
    class ConvNet_Dropout(general_model):
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
            conv_feat_size = get_size_of_conv_output(input_shape, self._forward_features)
            self.fc1 = linear_layer(conv_feat_size, n_hidden_fc1)
            self.fc_out = linear_layer(n_hidden_fc1, n_classes)


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
    #  ConvNet for Omniglot
    # -------------------------------------------------------------------------------------------
    # based on https://github.com/katerakelly/pytorch-maml/blob/master/src/omniglot_net.py
    class OmniglotNet(general_model):
        def __init__(self):
            super(OmniglotNet, self).__init__()
            self.model_type = model_type
            self.model_name = model_name
            n_filt1 = 64  # 64
            n_filt2 = 64  # 64
            n_filt3 = 64  # 64

            self.conv1 = conv2d_layer(color_channels, n_filt1, kernel_size=3)
            self.bn1 = nn.BatchNorm2d(n_filt1, momentum=1, affine=True)
            # self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = conv2d_layer(n_filt1, n_filt2, kernel_size=3)
            self.bn2 = nn.BatchNorm2d(n_filt2, momentum=1, affine=True)
            # self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = conv2d_layer(n_filt2, n_filt3, kernel_size=3)
            self.bn3 = nn.BatchNorm2d(n_filt3, momentum=1, affine=True)
            # self.relu3 = nn.ReLU(inplace=True)
            conv_feat_size = get_size_of_conv_output(input_shape, self._forward_features)
            self.fc_out = linear_layer(conv_feat_size, n_classes)

        def _forward_features(self, x):
            x = F.elu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.elu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.elu(self.bn3(self.conv3(x)))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            return x

        def forward(self, x):
            x = self._forward_features(x)
            x = x.view(x.size(0), -1)
            x = self.fc_out(x)
            return x

            # -------------------------------------------------------------------------------------------
            #  ConvNet for Omniglot
            # -------------------------------------------------------------------------------------------
            # based on https://github.com/katerakelly/pytorch-maml/blob/master/src/omniglot_net.py

    class Conv3(general_model):
        def __init__(self):
            super(Conv3, self).__init__()
            self.model_type = model_type
            self.model_name = model_name
            n_filt1 = 64  # 64
            n_filt2 = 64  # 64
            n_filt3 = 64  # 64

            self.conv1 = conv2d_layer(color_channels, n_filt1, kernel_size=3)
            self.conv2 = conv2d_layer(n_filt1, n_filt2, kernel_size=3)
            self.conv3 = conv2d_layer(n_filt2, n_filt3, kernel_size=3)
            conv_feat_size = get_size_of_conv_output(input_shape, self._forward_features)
            self.fc_out = linear_layer(conv_feat_size, n_classes)

        def _forward_features(self, x):
            x = F.elu((self.conv1(x)))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.elu((self.conv2(x)))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.elu((self.conv3(x)))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            return x

        def forward(self, x):
            x = self._forward_features(x)
            x = x.view(x.size(0), -1)
            x = self.fc_out(x)
            return x

    # -------------------------------------------------------------------------------------------
    #  Return selected model:
    # -------------------------------------------------------------------------------------------
    if model_type == 'Standard':
        DenseNet = get_densenet_model_class(prm, input_channels=color_channels)
    else:
        DenseNet = get_bayes_densenet_model_class(prm, input_channels=color_channels)

    if model_name == 'FcNet2':
        model = FcNet2()
    elif model_name == 'FcNet3':
        model = FcNet3()
    elif model_name == 'ConvNet':
        model = ConvNet()
    elif model_name == 'ConvNet_Dropout':
        model = ConvNet_Dropout()
    elif model_name == 'OmniglotNet':
        model = OmniglotNet()
    elif model_name == 'Conv3':
        model = Conv3()

    elif model_name == 'DenseNet20':
        model = DenseNet(depth=20, num_classes=n_classes)
        model.model_type = model_type
        model.model_name = model_name

    elif model_name == 'DenseNet':
        model = DenseNet(depth=40, num_classes=n_classes)
        model.model_type = model_type
        model.model_name = model_name

    elif model_name == 'DenseNet60':
        model = DenseNet(depth=60, num_classes=n_classes)
        model.model_type = model_type
        model.model_name = model_name

    elif model_name == 'DenseNet100':
        model = DenseNet(depth=100, num_classes=n_classes)
        model.model_type = model_type
        model.model_name = model_name

    else:
        raise ValueError('Invalid model_name')

    if init_override:
        print('Initializing model with override values: {}'.format(init_override))
        for param in model.parameters():
            param.data.normal_(mean=init_override['mean'], std=init_override['std'])


    model.cuda()

    return model