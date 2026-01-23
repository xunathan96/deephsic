import math
import numpy as np
import torch
from tqdm import tqdm
from kernel import Kernel
__all__ = ['mmd2', 'permutation_test', 'test_power']


def mmd2_depreciated(k: Kernel,
         X: torch.Tensor,
         Y: torch.Tensor,
         statistic='u',
         onesampleU=True,
         compute_var=True,):
    r"""computes the MMD^2 of samples Xi~P and Yi~Q based on the specified (u or v)-statistics.
    X: (M, Dx) torch.Tensor
    Y: (N, Dy) torch.Tensor
    returns the scalar MMD^2 estimator."""

    Kxx: torch.Tensor = k(X,X)    # (M,M) gram matrix
    Kyy: torch.Tensor = k(Y,Y)    # (N,N) gram matrix
    Kxy: torch.Tensor = k(X,Y)    # (M,N) gram matrix
    m = Kxx.shape[-1]
    n = Kyy.shape[-1]

    ones_m = torch.ones_like(Kxx)
    ones_n = torch.ones_like(Kyy)
    eye_m = torch.eye(m, device=Kxx.device)
    eye_n = torch.eye(n, device=Kyy.device)
    mask_m = ones_m - eye_m
    mask_n = ones_n - eye_n
    Kxx_hol = mask_m*Kxx  # hollow matrix (diagonal zeroes)
    Kyy_hol = mask_n*Kyy

    if statistic=='u':
        if onesampleU:
            if not m==n:
                raise Exception(f'Expected X and Y to have the same shape but got shapes {X.shape} and {Y.shape}.')
            ones = torch.ones_like(Kxx)
            eye = torch.eye(n, device=Kxx.device)
            mask = ones - eye

            Hzz = Kxx + Kyy - 2*Kxy
            nP2 = n*(n-1)
            mmd2_est = (1/nP2)*(mask*Hzz).sum()
        else:
            ones_m = torch.ones_like(Kxx)
            ones_n = torch.ones_like(Kyy)
            eye_m = torch.eye(m, device=Kxx.device)
            eye_n = torch.eye(n, device=Kyy.device)
            mask_m = ones_m - eye_m
            mask_n = ones_n - eye_n

            mP2 = m*(m-1)
            nP2 = n*(n-1)
            mmd2_est = (1/mP2)*(Kxx_hol).sum() - (2/(m*n))*Kxy.sum() + (1/nP2)*(Kyy_hol).sum()

    elif statistic=='v':
        mmd2_est = (1/m**2)*Kxx.sum() - (2/(m*n))*Kxy.sum() + (1/n**2)*Kyy.sum()

    var_est = None
    if compute_var:
        # compute v-stat estimator of the variance of MMD^2
        Hzz = Kxx + Kyy - 2*Kxy
        Eh1 = (Hzz.sum(dim=-1)**2).sum(dim=-1)
        Eh2 = Hzz.sum(dim=(-1,-2))
        var_est = 4 * ((1/(m*n**2))*Eh1 - (1/(m*n)**2)*Eh2**2)
        var_est = torch.clamp(var_est, min=0)
    return mmd2_est, var_est



def mmd2(k: Kernel,
         X: torch.Tensor,
         Y: torch.Tensor,
         statistic='u',
         onesampleU=True,
         compute_var=True,):
    r"""computes the MMD^2 of samples Xi~P and Yi~Q based on the specified (u or v)-statistics.
    X: (M, Dx) torch.Tensor
    Y: (N, Dy) torch.Tensor
    returns the scalar MMD^2 estimator and variance."""
    Kxx = k(X,X)    # (M,M) gram matrix
    Kyy = k(Y,Y)    # (N,N) gram matrix
    Kxy = k(X,Y)    # (M,N) gram matrix
    return mmd2_fast(Kxx,
                     Kyy,
                     Kxy,
                     statistic=statistic,
                     onesampleU=onesampleU,
                     compute_var=compute_var)


def mmd2_fast(Kxx: torch.Tensor,
              Kyy: torch.Tensor,
              Kxy: torch.Tensor,
              statistic='u',
              onesampleU=True,
              compute_var=True,):
    r"""computes the MMD^2 of samples Xi~P and Yi~Q based on the specified (u or v)-statistics.
    Kxx: (M, M) gram matrix k(xi,xj)
    Kyy: (N, N) gram matrix k(yi,yj)
    Kxy: (M, N) gram matrix k(xi,yj)
    returns the scalar MMD^2 estimator and variance."""
    m = Kxx.shape[-1]
    n = Kyy.shape[-1]

    if statistic=='u':
        if onesampleU:
            if not m==n:
                raise Exception(f'Expected X and Y to have the same shape but got shapes {m} and {n}.')
            Hzz = Kxx + Kyy - Kxy - Kxy.T
            Hzz_hol = Hzz.fill_diagonal_(0) # modifies in-place
            mP2 = m*(m-1)
            mmd2_est = (1/mP2)*Hzz_hol.sum()
        else:
            Kxx_hol = Kxx.fill_diagonal_(0) # modifies in-place
            Kyy_hol = Kyy.fill_diagonal_(0)
            mP2 = m*(m-1)
            nP2 = n*(n-1)
            mmd2_est = (1/mP2)*Kxx_hol.sum() - (2/(m*n))*Kxy.sum() + (1/nP2)*Kyy_hol.sum()

    elif statistic=='v':
        mmd2_est = (1/m**2)*Kxx.sum() - (2/(m*n))*Kxy.sum() + (1/n**2)*Kyy.sum()

    var_est = None
    if compute_var:
        # compute variance estimator for unbiased one-sample MMD^2
        Hzz = Kxx + Kyy - Kxy - Kxy.T
        var_est = (4/(n**3))*(Hzz.sum(dim=-1)**2).sum() - (4/(n**4))*(Hzz.sum()**2)
        var_est = torch.clamp(var_est, min=0)

    return mmd2_est, var_est


def permutation_test(k: Kernel,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     compute_var: bool = False,
                     n_permutations: int = 500,
                     significance: float = 0.05,
                     test: str = 'two-sample'):
    if test == 'two-sample':
        return permutation_test_twosample(k, X, Y, compute_var, n_permutations, significance)
    elif test == 'independence':
        return permutation_test_independence(k, X, Y, compute_var, n_permutations, significance)
    elif test == 'split-independence':
        return permutation_test_split_independece(k, X, Y, compute_var, n_permutations, significance)
    else:
        raise NotImplementedError()

def permutation_test_twosample(k: Kernel,
                               X: torch.Tensor,
                               Y: torch.Tensor,
                               compute_var: bool = False,
                               n_permutations: int = 500,
                               significance: float = 0.05,):
    n = X.shape[0]
    device = X.device
    Kxx: torch.Tensor = k(X,X)    # (N,N) gram matrix
    Kyy: torch.Tensor = k(Y,Y)    # (N,N) gram matrix
    Kxy: torch.Tensor = k(X,Y)    # (N,N) gram matrix
    # compute the gram matrix Kzz for Z=(X,Y)
    Kxxy = torch.cat((Kxx, Kxy), dim=-1)    # (N,2N)
    Kyxy = torch.cat((Kxy.T, Kyy), dim=-1)  # (N,2N)
    Kxyxy = torch.cat((Kxxy, Kyxy), dim=0)  # (2N,2N)
    mmd2_est, var_est = mmd2_fast(Kxx, Kyy, Kxy, compute_var=compute_var)

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples Z=(X,Y) for null hypothesis (i.e. P=Q)
        shuffle_idx = torch.randperm(2*n, device=device)
        idx_x = shuffle_idx[:n]
        idx_y = shuffle_idx[n:]
        Kxx = Kxyxy[torch.meshgrid(idx_x, idx_x, indexing='ij')]
        Kyy = Kxyxy[torch.meshgrid(idx_y, idx_y, indexing='ij')]
        Kxy = Kxyxy[torch.meshgrid(idx_x, idx_y, indexing='ij')]
        mmd2_null,_ = mmd2_fast(Kxx, Kyy, Kxy, compute_var=False)
        stats.append(mmd2_null.item())
        if mmd2_null > mmd2_est:
            count += 1

    # compute p-value (prob of hsic, assuming the null hypothesis is true)
    p_value = count/n_permutations
    # compute rejection threshold r
    stats.sort()
    thresh = n*stats[int(n_permutations*(1-significance)//1)]   # NOTE: multiply by n since r is scaled by n
    return (mmd2_est.item(),
            var_est.item() if var_est is not None else None,
            p_value,
            thresh)


def permutation_test_independence(k: Kernel,
                                  X: torch.Tensor,
                                  Y: torch.Tensor,
                                  compute_var: bool = False,
                                  n_permutations: int = 500,
                                  significance: float = 0.05,):
    n = X.shape[0]
    device = X.device
    # compile samples Z=(X,Y) for the independence testing problem
    Y_shuff = Y[torch.randperm(n, device=device)]
    Z_null = (X, Y_shuff)    # null: Px*Py
    Z_alt = (X, Y)           # alternate: Pxy
    mmd2_est, var_est = mmd2(k, Z_null, Z_alt, compute_var=compute_var)

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples Y for null hypothesis (i.e. Pxy=Px*Py)
        perm = torch.randperm(n, device=device)
        Y00 = Y_shuff[perm]
        Y10 = Y[perm]
        Z_null = (X, Y00)
        Z_alt = (X, Y10)
        mmd2_null,_ = mmd2(k, Z_null, Z_alt, compute_var=False)
        stats.append(mmd2_null.item())
        if mmd2_null > mmd2_est:
            count += 1

    # compute p-value (prob of hsic, assuming the null hypothesis is true)
    p_value = count/n_permutations
    # compute rejection threshold r
    stats.sort()
    thresh = n*stats[int(n_permutations*(1-significance)//1)]   # NOTE: multiply by n since r is scaled by n
    return (mmd2_est.item(),
            var_est.item() if var_est is not None else None,
            p_value,
            thresh)



def permutation_test_independence_old(k: Kernel,
                                  X: torch.Tensor,
                                  Y: torch.Tensor,
                                  compute_var: bool = False,
                                  n_permutations: int = 500,
                                  significance: float = 0.05,):
    n = X.shape[0]
    device = X.device
    # compile samples Z=(X,Y) for the independence testing problem
    Y_shuff = Y[torch.randperm(n, device=device)]
    Z_null = (X, Y_shuff)    # null: Px*Py
    Z_alt = (X, Y)           # alternate: Pxy
    mmd2_est, var_est = mmd2(k, Z_null, Z_alt, compute_var=compute_var)

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples Y for null hypothesis (i.e. Pxy=Px*Py)
        Y_shuff = Y[torch.randperm(n, device=device)]
        Z_alt = (X, Y_shuff)
        mmd2_null,_ = mmd2(k, Z_null, Z_alt, compute_var=False)
        stats.append(mmd2_null.item())
        if mmd2_null > mmd2_est:
            count += 1

    # compute p-value (prob of hsic, assuming the null hypothesis is true)
    p_value = count/n_permutations
    # compute rejection threshold r
    stats.sort()
    thresh = n*stats[int(n_permutations*(1-significance)//1)]   # NOTE: multiply by n since r is scaled by n
    return (mmd2_est.item(),
            var_est.item() if var_est is not None else None,
            p_value,
            thresh)


def permutation_test_pairs(k: Kernel,
                           X: torch.Tensor,
                           Y: torch.Tensor,
                           compute_var: bool = False,
                           n_shuffles: int = 1,
                           n_permutations: int = 500,
                           significance: float = 0.05,):
    
    n = X.shape[0]
    device = X.device
    Z_null, Z_alt = compile_pairs(X, Y, n_shuffles)

    mmd2_est, var_est = mmd2(k, Z_null, Z_alt, onesampleU=False, compute_var=compute_var)

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples Y for null hypothesis (i.e. Pxy=Px*Py)
        Y_shuff = Y[torch.randperm(n, device=device)]
        Z_alt = (X, Y_shuff)
        mmd2_null,_ = mmd2(k, Z_null, Z_alt, onesampleU=False, compute_var=False)
        stats.append(mmd2_null.item())
        if mmd2_null > mmd2_est:
            count += 1

    # compute p-value (prob of hsic, assuming the null hypothesis is true)
    p_value = count/n_permutations
    # compute rejection threshold r
    stats.sort()
    thresh = n*stats[int(n_permutations*(1-significance)//1)]   # NOTE: multiply by n since r is scaled by n
    return (mmd2_est.item(),
            var_est.item() if var_est is not None else None,
            p_value,
            thresh)



def compile_pairs(X: torch.Tensor, Y: torch.Tensor, n_shuffles: int = 1,):
    # compile samples Z=(X,Y) for the independence testing problem
    n = X.shape[0]
    device = X.device
    assert n_shuffles < n, "n_shuffles must be less than n."

    shuffle_ind = list()
    base_shuffle = torch.randperm(n, device=device)
    for i in range(n_shuffles):
        shift_ind = (base_shuffle + i) % n
        shuffle_ind.append(shift_ind)
    shuffle_ind = torch.cat(shuffle_ind)    # (n * n_shuffles,)
    base_ind = torch.arange(n*n_shuffles) % n   # (n * n_shuffles,)

    Y_shuff = Y[shuffle_ind]
    X_shuff = X[base_ind]
    Z_null = (X_shuff, Y_shuff) # null: Px*Py
    Z_alt = (X, Y)              # alternate: Pxy
    return Z_null, Z_alt




def permutation_test_split_independece(k: Kernel,
                                       X: torch.Tensor,
                                       Y: torch.Tensor,
                                       compute_var: bool = False,
                                       n_permutations: int = 500,
                                       significance: float = 0.05,):
    n = X.shape[0]
    device = X.device
    # split joint samples X,Y ~ Pxy equally into Z_null and Z_alt
    null_idx = slice(n//2)      # :n//2
    alt_idx = slice(n//2, n)    # n//2:
    Y_split = Y[null_idx]
    Y_null = Y_split[torch.randperm(n//2, device=device)]
    Z_null = (X[null_idx], Y_null)      # null: Px*Py
    Z_alt = (X[alt_idx], Y[alt_idx])    # alternate: Pxy
    return permutation_test_twosample_multimodal(k, Z_null, Z_alt, compute_var, n_permutations, significance)


def permutation_test_twosample_multimodal(k: Kernel,
                                          X: tuple[torch.Tensor],
                                          Y: tuple[torch.Tensor],
                                          compute_var: bool = False,
                                          n_permutations: int = 500,
                                          significance: float = 0.05,):
    n = X[0].shape[0]
    device = X[0].device
    Kxx: torch.Tensor = k(X,X)    # (N,N) gram matrix
    Kyy: torch.Tensor = k(Y,Y)    # (N,N) gram matrix
    Kxy: torch.Tensor = k(X,Y)    # (N,N) gram matrix
    # compute the gram matrix Kzz for Z=(X,Y)
    Kxxy = torch.cat((Kxx, Kxy), dim=-1)    # (N,2N)
    Kyxy = torch.cat((Kxy.T, Kyy), dim=-1)  # (N,2N)
    Kxyxy = torch.cat((Kxxy, Kyxy), dim=0)  # (2N,2N)
    mmd2_est, var_est = mmd2_fast(Kxx, Kyy, Kxy, compute_var=compute_var)

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples Z=(X,Y) for null hypothesis (i.e. P=Q)
        shuffle_idx = torch.randperm(2*n, device=device)
        idx_x = shuffle_idx[:n]
        idx_y = shuffle_idx[n:]
        Kxx = Kxyxy[torch.meshgrid(idx_x, idx_x, indexing='ij')]
        Kyy = Kxyxy[torch.meshgrid(idx_y, idx_y, indexing='ij')]
        Kxy = Kxyxy[torch.meshgrid(idx_x, idx_y, indexing='ij')]
        mmd2_null,_ = mmd2_fast(Kxx, Kyy, Kxy, compute_var=False)
        stats.append(mmd2_null.item())
        if mmd2_null > mmd2_est:
            count += 1

    # compute p-value (prob of hsic, assuming the null hypothesis is true)
    p_value = count/n_permutations
    # compute rejection threshold r
    stats.sort()
    thresh = n*stats[int(n_permutations*(1-significance)//1)]   # NOTE: multiply by n since r is scaled by n
    return (mmd2_est.item(),
            var_est.item() if var_est is not None else None,
            p_value,
            thresh)




# ==============================
#       HELPER FUNCTIONS
# ==============================

def pDist2(X: np.ndarray, Y: np.ndarray):
    r"""compute all paired (squared) distances between samples of X and Y
    X: (Nx, D) np.array
    Y: (Ny: D) np.array
    Returns matrix of paired distances of size (Nx, Ny)"""
    xyT = X @ Y.T                       # (Nx, Ny) pairwise inner products <x_i, y_j>
    x_norm = np.sum(X**2, axis=-1)      # (Nx,)
    y_norm = np.sum(Y**2, axis=-1)      # (Ny,)
    y_norm = y_norm[np.newaxis, :]      # (1, Ny)
    pdist2 = x_norm - 2*xyT + y_norm    # (Nx, Ny) pairwise distances |x_i - y_j|^2
    pdist2[pdist2<0] = 0
    return pdist2

def normal_cdf(value):
    loc = 0.0
    scale = 1.0
    return 0.5 * (1 + torch.erf((value - loc) * (1/scale) / math.sqrt(2)))
