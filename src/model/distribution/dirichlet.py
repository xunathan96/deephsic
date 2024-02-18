__all__ = ['Dirichlet']
import torch
import torch.nn as nn
import distribution


class Dirichlet(nn.Module, distribution.Dirichlet):

    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        alpha = torch.empty(self.dim)
        nn.init.uniform_(alpha, a=-1, b=1)
        self.log_concentration = nn.Parameter(alpha)
        super(nn.Module, self).__init__(concentration=self.concentration)

    @property
    def concentration(self):
        return torch.exp(self.log_concentration)    # constraint: alpha > 0

    @concentration.setter
    def concentration(self, value: torch.Tensor):
        if torch.all(value <= 0):
            raise Exception(f'concentration must be positive, but got {value}.')
        self._concentration = value
