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



def pairscore(f: nn.Module,
              X: torch.Tensor,
              Y: torch.Tensor) -> torch.Tensor:
    # compute T - T0
    Fxy = gram(f, X, Y)
    var_est = torch.var(torch.diagonal(Fxy, offset=0, dim1=-2, dim2=-1), dim=-1)
    T_shift_est = pairscore_fast(Fxy)
    return T_shift_est, var_est

def pairscore_fast(Fxy: torch.Tensor) -> torch.Tensor:
    n = Fxy.size(0)
    trF = torch.trace(Fxy)
    sumF = torch.sum(Fxy)
    return (trF / (n-1)) - (sumF / (n*(n-1)))




def T(f: nn.Module,
            X: torch.Tensor,
            Y: torch.Tensor,):
    r"""Computes the T value for samples (Xi, Yi) ~ Pxy."""
    assert X.size(0) == Y.size(0)
    fxy = f(X,Y)  # (n,) vector of f(Xi,Yi)
    T_est = torch.mean(fxy)
    var_est = torch.var(fxy, dim=-1)
    return T_est, var_est



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
    Fxy = gram(f, X, Y)
    nce = pairscore_fast(Fxy)

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples in Y for null hypothesis (i.e. Px*Py)
        shuffle_idx = torch.randperm(n, device=device)
        Fxy_shuffled = Fxy[:, shuffle_idx]
        nce_null = pairscore_fast(Fxy_shuffled)
        stats.append(nce_null.item())
        if nce_null >= nce:
            count += 1

    p_value = count/n_permutations
    return nce.item(), p_value
