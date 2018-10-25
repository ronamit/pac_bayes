from torch.autograd import Variable
import torch
from torch.nn.modules.module import Module
from torch.nn.modules.container import Sequential
from torch.nn.modules.activation import LogSoftmax
from torch.nn import functional as F
import math


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class Logistic_Binary_Loss(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input `x` (a 2D mini-batch Tensor) and
    target `y` (which is a tensor containing either `1` or `-1`).

    ::

        loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()

    The normalization by the number of elements in the input can be disabled by
    setting `self.size_average` to ``False``.
    """

    def forward(self, input, target):
        _assert_no_grad(target)
        assert input.shape[1] == 1 # this loss works only for binary classification
        input = input[:, 0]
        # switch labels to {-1,1}
        target = target.float() * 2 - 1
        # return F.soft_margin_loss(input_, target_, size_average=self.size_average) / math.log(2)
        return torch.log(1 + torch.exp(-target * input)).mean() / math.log(2)

