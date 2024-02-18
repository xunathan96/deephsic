import torch
import torch.nn as nn
from abc import ABC, abstractmethod
__all__ = ['Kernel', 'BaseKernel']

class Kernel(ABC):

    def __call__(self, X: torch.Tensor, Y: torch.Tensor):
        return self.gram(X, Y)

    @abstractmethod
    def gram(self, X: torch.Tensor, Y: torch.Tensor):
        ...


class BaseKernel(nn.Module):

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        return self.gram(X, Y)

    @abstractmethod
    def gram(self, X: torch.Tensor, Y: torch.Tensor):
        ...

