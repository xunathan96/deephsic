import torch
import torch.nn as nn
from tqdm import tqdm

__all__ = ['T', 'permutation_test']



def gram(f: nn.Module,
         X: torch.Tensor,
         Y: torch.Tensor) -> torch.Tensor:
    r"""Computes the square gram matrix f(xi,yj) given batches X=(x1,...,xn) and Y=(y1,...,yn)."""
    if (n:=X.size(0)) != Y.size(0):
        raise Exception(f"Error: expected batches X and Y to have equal number of samples.")
    device = X.device
    Fxy = torch.empty((n,n), device=device)
    idx = torch.arange(n, device=device)
    idy = torch.arange(n, device=device)
    for i in range(n):
        shift_idy = idy - i
        Y_shift = Y[shift_idy]  # shift Y to right
        Fxy[idx, shift_idy] = f(X, Y_shift)
    return Fxy


def T(f: nn.Module,
            X: torch.Tensor,
            Y: torch.Tensor,):
    r"""Computes the T value for samples (Xi, Yi) ~ Pxy."""
    assert X.size(0) == Y.size(0)
    fxy = f(X,Y)  # (n,) vector of f(Xi,Yi)
    T_est = torch.mean(fxy)
    var_est = sample_var(fxy)
    return T_est, var_est


def sample_var(X: torch.Tensor) -> torch.Tensor:
    mu = X.mean(-1, keepdim=True)
    return torch.mean((X - mu)**2, dim=-1)


def permutation_test(f: nn.Module,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     n_permutations: int = 500):
    r"""perform a permutation test to compute the test statistic and p-value.
    f: function taking X and Y with scalar outputs
    X: (N,*,Dx) samples from Pxy
    Y: (N,*,Dy) samples from Pxy
    returns the test statistic and its p-value under the null hypothesis
    """
    n = X.shape[0]
    device = X.device
    Tstat, _ = T(f, X, Y)

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples in Y for null hypothesis (i.e. Px*Py)
        shuffle_idx = torch.randperm(n, device=device)

        Y_null = Y[shuffle_idx]
        Tstat_null, _ = T(f, X, Y_null)

        stats.append(Tstat_null.item())
        if Tstat_null >= Tstat:
            count += 1

    p_value = count/n_permutations
    return Tstat.item(), p_value
