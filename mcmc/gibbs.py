import torch
from .base import MCMC
from distribution import Distribution

class Gibbs(MCMC):

    def __init__(self, proposal: Distribution):
        super().__init__()
        self.proposal = proposal

    def step(self, X: torch.Tensor):
        r"""computes one update of Gibbs sampling given X of size (N, D)."""
        dim = X.shape[-1]
        for j in range(dim):
            Xj = self.proposal.sample(X, j)   # sample from the conditional p(xj|x!=j)
            X[:,j] = Xj
        return X

    def simulate(self, burn_in, X0):
        r"""generates samples via Gibbs algorithm with a given burn-in period
        and initial value X of size (N, D)."""
        Xn = X0
        for n in range(burn_in):
            Xn = self.step(Xn)
        return Xn




