__all__ = ['Gaussian']
import torch
import math
from .base import Distribution


class Gaussian(Distribution):
    
    def __init__(self,
                 mean: torch.Tensor,
                 cov: torch.Tensor):
        super().__init__()

        self.dim = len(mean)
        if not (cov.shape[-1]==cov.shape[-2]==self.dim):
            raise Exception(f'Expected cov to be a square positive-semi definite matrix.')
        self.mean = mean
        self.cov = cov
        self.log_det = torch.logdet(cov)
        if self.log_det==torch.nan:
            raise Exception(f'The covariance matrix must be positive semi-definite.')
        elif self.log_det==torch.inf:
            raise Exception(f'The covariance implies a degenerate distribution.')
        self.precision = torch.linalg.inv(cov)
        self.L = torch.linalg.cholesky(cov)


    def sample(self, n_samples=1) -> torch.Tensor:
        X = torch.randn((n_samples, self.dim)).to(self.L.device)   # (N, D)
        return X @ self.L.T + self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        #mahalanobis = -0.5 * torch.diagonal((x-self.mean) @ self.precision @ (x-self.mean).T)
        mahalanobis = -0.5 * torch.einsum('...i,ij,...j->...', x-self.mean, self.precision, x-self.mean)
        log2pi = math.log(2*torch.pi)
        return mahalanobis - 0.5*self.dim*log2pi - 0.5*self.log_det

    def score(self,
              x: torch.Tensor,
              autograd: bool = False,
              retain_graph: bool = True) -> torch.Tensor:
        if not autograd:
            # explicit calculation of the score: ∇logp(x) = -Σ^{-1}(x-mu)
            return - (x-self.mean) @ self.precision
        else:
            return super().score(x, retain_graph)






def main():
    import matplotlib.pyplot as plt
    from torch.autograd import grad

    # define gaussian distribution
    mean = torch.Tensor([0,0])
    cov = torch.Tensor([
        [1, 0.8],
        [0.8, 1],
    ])
    gaussian = Gaussian(mean, cov)


    # compute log density at X
    X = torch.Tensor([
        [0,0],
        [1,1],
        [1,-1],
    ]).requires_grad_(True)
    logp = gaussian.log_prob(X)
    print(logp)
    """
    torch_gaussian = torchdistr.MultivariateNormal(mean, covariance_matrix=cov)
    logp = torch_gaussian.log_prob(X)
    print(logp)
    """

    # compute and compare autograd vs explicit score functions 
    score_X = gaussian.score(X)
    score2_X = grad(score_X.sum(), X, create_graph=False)[0]
    print(score_X)
    print(score2_X)

    score_X = gaussian.score(X, explicit=False)
    score2_X = grad(score_X.sum(), X, create_graph=False)[0]
    print(score_X)
    print(score2_X)

    # sample from gaussian
    x = gaussian.sample(n_samples=1000)
    plt.scatter(x[:,0], x[:,1], s=0.5)
    plt.axis('equal')
    plt.show()
