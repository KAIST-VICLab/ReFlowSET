"""STUB for vision_aided_loss (the CLIP discriminator), stage 1 ONLY.
E3Diff's model.py instantiates the discriminator unconditionally, but only USES it
when stage==2 with lambda_gan>0, so stage 1 (lambda_gan=0) can import this
stub instead of downloading CLIP weights.

DO NOT let this shadow the real package when training stage 2 (../e3diff/):
it returns a constant, so the GAN loss becomes constant and the run is not the
authors' method. e3diff/run.sh asserts on this; for real stage-2 runs,
install the real vision-aided-loss package.
Needs >=1 parameter because model.py builds an Adam over net_disc.parameters().
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, cv_type='clip', loss_type='multilevel_sigmoid_s', device='cuda'):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.cv_ensemble = nn.Module()

    def forward(self, x, for_G=False, for_real=False):
        return self.dummy.expand(x.shape[0])
