import torch
import torch.nn as nn
from .base import BaseKernel
__all__ = ['Polynomial', 'Linear']


class Polynomial(BaseKernel):
    def __init__(self,
                 shift: float = 0.,
                 degree: int = 1,
                 trainable: bool = False):
        super().__init__()
        self.degree = degree
        shift_raw = torch.empty(1)
        if trainable:
            nn.init.normal_(shift_raw, mean=shift, std=1)
        else:
            nn.init.constant_(shift_raw, val=shift)
        self.shift = nn.Parameter(shift_raw, requires_grad=trainable)

    def gram(self, X: torch.Tensor, Y: torch.Tensor):
        r"""compute the kernel gram matrix between samples X and Y
        X: (Nx, D) torch.Tensor
        Y: (Ny, D) torch.Tensor
        returns the gram matrix of size (Nx, Ny) with elements k(x_i, y_j)"""
        return (X @ Y.T + self.shift)**self.degree


class Linear(Polynomial):
    def __init__(self, shift, trainable):
        super().__init__(shift=shift, degree=1, trainable=trainable)

    #def gram(self, X: torch.Tensor, Y: torch.Tensor):
    #    return X @ Y.T
