import torch
from abc import ABC, abstractmethod
from torch.autograd import grad


class Distribution(ABC):

    @abstractmethod
    def log_prob(self,
                 x: torch.Tensor) -> torch.Tensor:
        r"""computes the log probabilities of the given sample x of size (N, D)"""

    @abstractmethod
    def sample(self,
               n_samples: int = 1) -> torch.Tensor:
        ...

    def score(self,
              x: torch.Tensor,
              retain_graph: bool = True) -> torch.Tensor:
        r"""computes the score function (gradient log-density) of the distribution wrt x of size (N, D).
        If retain_graph is True, then the returned score keeps its computation graph allowing for backpropagation."""
        REQUIRE_GRAD = x.requires_grad
        x.requires_grad_(True)
        logprob = self.log_prob(x)
        score = grad(logprob.sum(), x, create_graph=retain_graph)[0]
        x.requires_grad_(REQUIRE_GRAD)
        return score

