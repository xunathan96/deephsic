import torch
from .base import Kernel
__all__ = ['Polynomial', 'Linear']


class Polynomial(Kernel):
    def __init__(self,
                 shift: float = 0.,
                 degree: int = 1,):
        self.shift = shift
        self.degree = degree
    
    def gram(self, X: torch.Tensor, Y: torch.Tensor):
        r"""compute the kernel gram matrix between samples X and Y
        X: (Nx, D) torch.Tensor
        Y: (Ny, D) torch.Tensor
        returns the gram matrix of size (Nx, Ny) with elements k(x_i, y_j)"""
        return (X @ Y.T + self.shift)**self.degree


class Linear(Polynomial):
    def __init__(self):
        super().__init__(shift=0, degree=1)

    def gram(self, X: torch.Tensor, Y: torch.Tensor):
        return X @ Y.T
