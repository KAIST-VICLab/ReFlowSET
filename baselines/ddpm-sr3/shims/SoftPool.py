"""Pure-PyTorch drop-in for the SoftPool CUDA extension (only SoftPool2d / soft_pool2d
are used by E3Diff, in model/sr3_modules/unet.py CPEN, kernel 2x2 stride 2).
softpool(x) = sum(exp(x)*x)/sum(exp(x)) over each window == avg_pool(x*e)/avg_pool(e).
Parameter-free, exact math, exact autograd. No build step needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_pool2d(x, kernel_size=2, stride=None):
    if stride is None:
        stride = kernel_size
    e = torch.exp(x)
    return F.avg_pool2d(x * e, kernel_size, stride) / F.avg_pool2d(e, kernel_size, stride).clamp_min(1e-12)


class SoftPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return soft_pool2d(x, self.kernel_size, self.stride)
