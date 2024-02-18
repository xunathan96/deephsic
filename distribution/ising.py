from .base import Distribution
import torch


class IsingProposal(Distribution):
    def __init__(self, n_rows, n_cols, temp=1):
        self.temp = temp
        self.n_rows = n_rows
        self.n_cols = n_cols

    def log_prob(self, X: torch.Tensor):
        return super().log_prob(X)

    def sample(self, X: torch.Tensor, idx: int):
        r"""sample from the conditional P(Xj|X!=j)"""
        n = X.shape[0]
        dim = X.shape[-1]
        if not self.n_cols*self.n_rows == dim:
            raise Exception(f'Expected array of size {self.n_cols*self.n_rows} but found size {dim}.')

        # reshape X and idx into its matrix form
        Xmat = X.view(-1, self.n_rows, self.n_cols)         # (N, R, C)
        i,j = int(idx/self.n_cols), int(idx%self.n_cols)

        s = 0
        if i-1>=0:
            s += Xmat[:,i-1,j]
        if i+1<self.n_rows:
            s += Xmat[:,i+1,j]
        if j-1>=0:
            s += Xmat[:,i,j-1]
        if j+1<self.n_cols:
            s += Xmat[:,i,j+1]

        thresh = 1/(1+torch.exp(-2*s/self.temp))    # (N,)
        p = torch.rand(n)                           # (N,)
        samples = torch.ones_like(p)
        samples[p>=thresh] = -1
        return samples

        Xmat[p<thresh, i, j] = 1
        Xmat[p>=thresh, i, j] = -1
        return Xmat.view(-1, self.n_rows*self.n_cols)
