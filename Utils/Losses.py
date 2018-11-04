from __future__ import absolute_import, division, print_function

import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import math

# -----------------------------------------------------------------------------------------------------------#
# Returns loss function
# -----------------------------------------------------------------------------------------------------------#
def get_loss_func(loss_type):
    # Note: 1. the loss function use the un-normalized net outputs (scores, not probabilities)
    #       2. The returned loss is summed (not averaged) over samples!!!

    if loss_type == 'CrossEntropy':
        return nn.CrossEntropyLoss(reduction='sum').cuda()

    elif loss_type == 'L2_SVM':
        return nn.MultiMarginLoss(p=2, margin=1, weight=None, reduction='sum').cuda()

    elif loss_type == 'Logistic_binary':
        return Logistic_Binary_Loss(reduction='sum').cuda()

    elif loss_type == 'Logistic_Binary_Clipped':
        return Logistic_Binary_Loss_Clipped(reduction='sum').cuda()


    elif loss_type == 'Zero_One':
        return Zero_One_Loss(reduction='sum').cuda()

    else:
        raise ValueError('Invalid loss_type')



# -----------------------------------------------------------------------------------------------------------#
# Definitions of loss functions
# -----------------------------------------------------------------------------------------------------------#

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"
# -----------------------------------------------------------------------------------------------------------#
# Base class for loss functions
class _Loss(Module):
    def __init__(self, reduction='sum'):
        super(_Loss, self).__init__()
        self.reduction = reduction

# -----------------------------------------------------------------------------------------------------------#

class Logistic_Binary_Loss(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input `x` (a 2D mini-batch Tensor) and
    target `y` (which is a tensor containing either `1` or `-1`).

    ::

        loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()  / math.log(2)

    The normalization by the number of elements in the input can be disabled by
    setting `self.size_average` to ``False``.
    """

    def forward(self, input, target):
        # validity checks
        _assert_no_grad(target)
        assert input.shape[1] == 1 # this loss works only for binary classification
        input = input[:, 0]
        assert self.reduction == 'sum'
        # switch labels to {-1,1}
        target = target.float() * 2 - 1
        loss_vec = torch.log(1 + torch.exp(-target * input)) / math.log(2)
        loss_sum = loss_vec.sum()
        return loss_sum
# -----------------------------------------------------------------------------------------------------------#

class Logistic_Binary_Loss_Clipped(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input `x` (a 2D mini-batch Tensor) and
    target `y` (which is a tensor containing either `1` or `-1`).

    ::

        loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()  / math.log(2)

    The normalization by the number of elements in the input can be disabled by
    setting `self.size_average` to ``False``.
    """

    def forward(self, input, target):
        # validity checks
        _assert_no_grad(target)
        assert input.shape[1] == 1 # this loss works only for binary classification
        input = input[:, 0]
        assert self.reduction == 'sum'
        # switch labels to {-1,1}
        target = target.float() * 2 - 1
        loss_vec = torch.log(1 + torch.exp(-target * input)) / math.log(2)
        # clamp\clipp values to [0,1]:
        loss_vec = loss_vec.clamp(0, 1)
        loss_sum = loss_vec.sum()
        return loss_sum

# -----------------------------------------------------------------------------------------------------------#

class Zero_One_Loss(_Loss):
    # zero one-loss of binaty classifier with labels {-1,1}
    def forward(self, input, target):
        # validity checks
        _assert_no_grad(target)
        assert input.shape[1] == 1 # this loss works only for binary classification
        input = input[:, 0]
        assert self.reduction == 'sum'
        # switch labels to {-1,1}
        target = target.float() * 2 - 1
        # loss_sum =  F.soft_margin_loss(input_, target_, size_average=self.size_average) / math.log(2)
        loss_sum = (target != torch.sign(input)).sum().float()
        return loss_sum

# -----------------------------------------------------------------------------------------------------------#
