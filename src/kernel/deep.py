import torch
import torch.nn as nn
import math
from .base import Kernel, BaseKernel
__all__ = ['DeepKernel']


class DeepKernel(BaseKernel):
    def __init__(self,
                 featurizer: nn.Module,
                 feature_kernel: Kernel,
                 smoothing_kernel: Kernel,
                 eps: float = 0.1,
                 trainable: bool = False):
        super().__init__()
        if not 0<=eps<=1:
            raise Exception(f'Expected eps to be between 0 and 1 but got {eps}')
        self.featurizer = featurizer
        self.feature_kernel = feature_kernel
        self.smoothing_kernel = smoothing_kernel
        raw_eps = torch.empty(1)
        if trainable:
            nn.init.normal_(raw_eps, mean=-math.log((1-eps)/eps), std=1)
        else:
            nn.init.constant_(raw_eps, val=-math.log((1-eps)/eps))
        self.raw_eps = nn.Parameter(raw_eps, requires_grad=trainable)

    @property
    def eps(self):
        return torch.sigmoid(self.raw_eps)

    def gram(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        r"""compute the kernel gram matrix between samples X and Y of the same shape
        X: (Nx, *) torch.Tensor
        Y: (Ny, *) torch.Tensor
        returns the gram matrix of size (Nx, Ny) with elements k(x_i, y_j)"""
        phi_X = self.featurizer(X)  # (Nx, D)
        phi_Y = self.featurizer(Y)  # (Ny, D)
        if X.dim() > 2: X = torch.flatten(X, start_dim=1)   # (Nx, d)
        if Y.dim() > 2: Y = torch.flatten(Y, start_dim=1)   # (Ny, d)
        Kxy = self.feature_kernel(phi_X, phi_Y) # (Nx, Ny)
        Qxy = self.smoothing_kernel(X, Y)       # (Nx, Ny)
        return (1-self.eps)*Kxy + self.eps*Qxy

