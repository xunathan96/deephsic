import math
import torch
from .base import MCMC
from distribution import Distribution


class MALA(MCMC):

    def __init__(self,
                 stationary_distribution: Distribution,
                 step_size: float = 0.01,
                 x_min: float = None,
                 x_max: float = None):
        super().__init__()
        self.pi = stationary_distribution
        self.tau = step_size
        self.x_min = x_min
        self.x_max = x_max
        if self.tau <= 0:
            raise Exception(f'Step size must be positive, but got a step size of {step_size}.')

    def step(self,
             x: torch.Tensor,
             retain_graph: bool = False):
        r"""computes one update of MALA given x of size (N, D).
        If retain_graph is True, then the returned sample keeps its computation graph."""
        eps = torch.randn_like(x)
        xN = x + 0.5*self.tau*self.pi.score(x, retain_graph=retain_graph) + math.sqrt(self.tau)*eps
        if not ((self.x_min is None) and (self.x_max is None)):
            xN = torch.clamp(xN, min=self.x_min, max=self.x_max)
        if retain_graph:
            return xN
        return xN.detach()



