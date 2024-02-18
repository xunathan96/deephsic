import math
import numpy as np
import torch
from tqdm import tqdm
from kernel import Kernel
__all__ = ['hsic', 'permutation_test', 'test_power']


def hsic_depreciated(k: Kernel,
         l: Kernel,
         X: torch.Tensor,
         Y: torch.Tensor,
         statistic='u',
         onesampleU=True,
         compute_var=True,):
    r"""computes the HSIC of samples (Xi, Yi) ~ Pxy based on the specified (u or v)-statistics.
    X: (N, Dx) torch.Tensor
    Y: (N, Dy) torch.Tensor
    returns the scalar HSIC estimator."""
    Kxx = k(X, X)   # (N, N) gram matrix
    Lyy = l(Y, Y)   # (N, N) gram matrix
    n = Kxx.shape[-1]
    ones = torch.ones_like(Kxx)
    eye = torch.eye(n, device=Kxx.device)
    mask = ones - eye
    Kxx_hol = mask*Kxx  # hollow matrix (diagonal zeroes)
    Lyy_hol = mask*Lyy

    if statistic=='u':
        if onesampleU:
            trKL = torch.einsum('ij,ji', Kxx_hol, Lyy_hol)
            trKL1 = torch.einsum('ij,jk->', Kxx_hol, Lyy_hol)
            trK1L1 = torch.einsum('ij,kl->', Kxx_hol, Lyy_hol)
            #trKL = torch.einsum('ij,ji', Kxx_hol, Lyy_hol)
            #trKL1 = torch.einsum('ij,jk,ki', Kxx_hol, Lyy_hol, ones)
            #trK1L1 = torch.einsum('ij,jk,kl,li', Kxx_hol, ones, Lyy_hol, ones)
            c1 = n*(n-3)
            c2 = n-2
            c3 = (n-1)*(n-2)
            hsic_est = (1/c1)*(trKL - (2/c2)*trKL1 + (1/c3)*trK1L1)
        else:
            trKL = torch.einsum('ij,ij->', Kxx, Lyy)
            shift = torch.einsum('ii,ii->', Kxx, Lyy)       # i=j
            trKL -= shift

            trKL1 = torch.einsum('ij,ik->', Kxx, Lyy)
            shift = (torch.einsum('ii,ik->', Kxx, Lyy)      # i=j
                    + torch.einsum('ij,ii->', Kxx, Lyy)    # i=k
                    + torch.einsum('ij,ij->', Kxx, Lyy)    # j=k
                    - 2*torch.einsum('ii,ii->', Kxx, Lyy)) # i=j=k
            trKL1 -= shift

            trK1L1 = torch.einsum('ij,kl->', Kxx, Lyy)
            shift = (torch.einsum('ii,kl->', Kxx, Lyy)         # i=j
                    + torch.einsum('ij,il->', Kxx, Lyy)        # i=k
                    + torch.einsum('ij,ki->', Kxx, Lyy)        # i=l
                    + torch.einsum('ij,jl->', Kxx, Lyy)        # j=k
                    + torch.einsum('ij,kj->', Kxx, Lyy)        # j=l
                    + torch.einsum('ij,kk->', Kxx, Lyy)        # k=l
                    - 5*torch.einsum('ii,il->', Kxx, Lyy)      # i=j=k
                    - 5*torch.einsum('ii,ki->', Kxx, Lyy)      # i=j=l
                    - 5*torch.einsum('ij,ii->', Kxx, Lyy)      # i=k=l
                    - 5*torch.einsum('ij,jj->', Kxx, Lyy)      # j=k=l
                    + 15*torch.einsum('ii,ii->', Kxx, Lyy))    # i=j=k=l
            trK1L1 -= shift

            nP2 = n*(n-1)
            nP3 = n*(n-1)*(n-2)
            nP4 = n*(n-1)*(n-2)*(n-3)
            hsic_est = (1/nP2)*trKL - (2/nP3)*trKL1 + (1/nP4)*trK1L1

    elif statistic=='v':
        trKL = torch.einsum('ij,ij', Kxx, Lyy)
        trKL1 = torch.einsum('ij,jk->', Kxx, Lyy)
        trK1L1 = torch.einsum('ij,kl->', Kxx, Lyy)
        hsic_est = (1/(n**2))*trKL - (2/(n**3))*trKL1 + (1/(n**4))*trK1L1
        # more explicit computation
        #H = eye - (1/n)*ones    # (N,N)
        #TrKHLH = torch.einsum('ij,jk,kl,li', Kxx, H, Lyy, H)
        #hsic_est = (1/n**2)*TrKHLH

    var_est = None
    if compute_var:
        Kxx1 = Kxx_hol.sum(dim=-1)
        Lyy1 = Lyy_hol.sum(dim=-1)
        z1 = torch.sum(Kxx_hol*Lyy_hol, dim=-1)
        z2 = torch.einsum('ij,ji', Kxx_hol, Lyy_hol) - Kxx_hol @ Lyy1 - Lyy_hol @ Kxx1
        z3 = Kxx1 * Lyy1
        z4 = Lyy1.sum()*Kxx1 + Kxx1.sum()*Lyy1 - torch.sum(Kxx_hol@Lyy_hol)
        h = ((n-2)**2)*z1 + (n-2)*z2 - n*z3 + z4

        nsub1P3 = (n-1)*(n-2)*(n-3)
        r = h @ h / (4.*n*(nsub1P3**2))
        var_est = (16/n)*(r - hsic_est**2)
        var_est = torch.clamp(var_est, min=0)
    return hsic_est, var_est


def hsic(k: Kernel,
         l: Kernel,
         X: torch.Tensor,
         Y: torch.Tensor,
         statistic='u',
         onesampleU=True,
         compute_var=True,):
    r"""computes the HSIC of samples (Xi, Yi) ~ Pxy based on the specified (u or v)-statistics.
    X: (N, Dx) torch.Tensor
    Y: (N, Dy) torch.Tensor
    returns the scalar HSIC estimator."""
    if not X.shape[0] == Y.shape[0]:
        raise Exception(f'Expected X and Y to have the same number of samples but got {X.shape[0]} and {Y.shape[0]}.')
    Kxx = k(X, X)   # (N, N) gram matrix
    Lyy = l(Y, Y)   # (N, N) gram matrix
    return hsic_fast(Kxx,
                     Lyy,
                     statistic,
                     onesampleU,
                     compute_var)


def hsic_fast(Kxx: torch.Tensor,
              Lyy: torch.Tensor,
              statistic='u',
              onesampleU=True,
              compute_var=True,):
    r"""computes the HSIC of samples (Xi, Yi) ~ Pxy based on the specified (u or v)-statistics.
    Kxx: (N, N) gram matrix k(xi,xj)
    Lyy: (N, n) gram matrix l(yi,yj)
    returns the scalar HSIC estimator."""
    if not Kxx.shape == Lyy.shape:
        raise Exception(f'Expected Kxx and Lyy to have the same shape but got shapes {Kxx.shape} and {Lyy.shape}.')

    n = Kxx.shape[-1]
    ones = torch.ones_like(Kxx)
    eye = torch.eye(n, device=Kxx.device)
    mask = ones - eye
    Kxx_hol = mask*Kxx  # hollow matrix (diagonal zeroes)
    Lyy_hol = mask*Lyy  # NOTE: torch.fill_diagonal_ will change in-place, which affects permutation test

    if statistic=='u':
        if onesampleU:
            trKL = torch.einsum('ij,ji', Kxx_hol, Lyy_hol)
            trKL1 = torch.einsum('ij,jk->', Kxx_hol, Lyy_hol)
            trK1L1 = torch.einsum('ij,kl->', Kxx_hol, Lyy_hol)
            c1 = n*(n-3)
            c2 = n-2
            c3 = (n-1)*(n-2)
            hsic_est = (1/c1)*(trKL - (2/c2)*trKL1 + (1/c3)*trK1L1)
        else:
            ...

    elif statistic=='v':
        trKL = torch.einsum('ij,ij', Kxx, Lyy)
        trKL1 = torch.einsum('ij,jk->', Kxx, Lyy)
        trK1L1 = torch.einsum('ij,kl->', Kxx, Lyy)
        hsic_est = (1/(n**2))*trKL - (2/(n**3))*trKL1 + (1/(n**4))*trK1L1

    var_est = None
    if compute_var:
        # compute variance estimator for unbiased one-sample HSIC
        Kxx1 = Kxx_hol.sum(dim=-1)
        Lyy1 = Lyy_hol.sum(dim=-1)
        z1 = torch.sum(Kxx_hol*Lyy_hol, dim=-1)
        z2 = torch.einsum('ij,ji', Kxx_hol, Lyy_hol) - Kxx_hol @ Lyy1 - Lyy_hol @ Kxx1
        z3 = Kxx1 * Lyy1
        z4 = Lyy1.sum()*Kxx1 + Kxx1.sum()*Lyy1 - torch.sum(Kxx_hol@Lyy_hol)
        h = ((n-2)**2)*z1 + (n-2)*z2 - n*z3 + z4

        nsub1P3 = (n-1)*(n-2)*(n-3)
        r = h @ h / (4.*n*(nsub1P3**2))
        var_est = 16*(r - hsic_est**2)      # (16/n)*(r - hsic_est**2)
        var_est = torch.clamp(var_est, min=0)
    return hsic_est, var_est



def permutation_test(k: Kernel,
                     l: Kernel,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     compute_var: bool = False,
                     n_permutations: int = 500,
                     significance: float = 0.05,):
    n = X.shape[0]
    device = X.device
    Kxx = k(X, X)   # (N, N) gram matrix
    Lyy = l(Y, Y)   # (N, N) gram matrix
    hsic, var = hsic_fast(Kxx, Lyy, compute_var=compute_var)    # NOTE: h potentially overflows when num samples N is large

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples in Y for null hypothesis (i.e. Pxy = Px*Py)
        #shuffle_idx = np.random.permutation(n)
        #Lyy_perm = Lyy[np.ix_(shuffle_idx, shuffle_idx)]
        shuffle_idx = torch.randperm(n, device=device)
        Lyy_perm = Lyy[torch.meshgrid(shuffle_idx, shuffle_idx, indexing='ij')]
        hsic_null,_ = hsic_fast(Kxx, Lyy_perm, compute_var=False)
        stats.append(hsic_null.item())
        if hsic_null > hsic:
            count += 1

    # compute p-value (prob of hsic, assuming the null hypothesis is true)
    p_value = count/n_permutations
    # compute rejection threshold r
    stats.sort()
    thresh = n*stats[int(n_permutations*(1-significance)//1)]   # NOTE: multiply by n since r is scaled by n
    return (hsic.item(),
            var.item() if var is not None else None,
            p_value,
            thresh)


def test_power(hsic: torch.Tensor,
               hsic_std: torch.Tensor,
               n: int,
               thresh: float,):
    # compute the asymptotic test power
    sqrtn = math.sqrt(n)
    quant = (sqrtn*hsic - thresh/sqrtn) / hsic_std
    power = normal_cdf(quant)
    return power


# ==============================
#       HELPER FUNCTIONS
# ==============================

def normal_cdf(value):
    loc = 0.0
    scale = 1.0
    return 0.5 * (1 + torch.erf((value - loc) * (1/scale) / math.sqrt(2)))
