__all__ = ['Dirichlet']
from .base import Distribution
import torch
from torch.distributions import Dirichlet as torchDirichlet


class Dirichlet(torchDirichlet, Distribution):

    def sample(self, n_samples: int) -> torch.Tensor:
        return super().sample((n_samples,))

    def score(self,
              x: torch.Tensor,
              retain_graph: bool = True) -> torch.Tensor:
        r"""computes the score of x, where x is the (unnormalized) sequence of xi with size (N, D-1)"""
        xd = 1-x.sum(dim=-1, keepdim=True)  # (N, 1)
        x = torch.cat((x,xd), dim=-1)       # (N, D)
        return super().score(x, retain_graph)[:,:-1]    # (N, D-1)
