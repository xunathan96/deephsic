import math
import torch
import torch.nn as nn
from tqdm import tqdm
__all__ = ['infoNCE', 'nwj', 'permutation_test']


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


def nwj(f: nn.Module,
        X: torch.Tensor,
        Y: torch.Tensor,):
    r"""Computes the NWJ metric for samples (Xi, Yi) ~ Pxy."""
    Fxy = gram(f, X, Y)
    return nwj_fast(Fxy)

def nwj_fast(Fxy: torch.Tensor):
    n = Fxy.size(0)
    trace = torch.trace(Fxy)
    return trace/n - math.exp(-1) * Fxy.exp().mean()


def permutation_test(f: nn.Module,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     n_permutations: int = 500):
    r"""perform a permutation test to compute the test statistic and p-value.
    f: function taking X and Y with scalar outputs
    X: (N,*,Dx) samples from Pxy
    Y: (N,*,Dy) samples from Pxy
    returns the test statistic value and its p-value under the null hypothesis
    """
    n = X.shape[0]
    device = X.device
    Fxy = gram(f, X, Y)
    stat = nwj_fast(Fxy)

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples in Y for null hypothesis (i.e. Px*Py)
        shuffle_idx = torch.randperm(n, device=device)
        Fxy_shuffled = Fxy[:, shuffle_idx]
        stat_null = nwj_fast(Fxy_shuffled)
        stats.append(stat_null.item())
        if stat_null >= stat:
            count += 1

    p_value = count/n_permutations
    return stat.item(), p_value




def main():
    class DummyNet(nn.Module):
        def forward(self, X, Y):
            return X + X*Y + Y**2

    f = DummyNet()
    # X = torch.arange(5)+0.0
    # Y = torch.arange(5)+0.0
    X = torch.rand(5)
    Y = torch.rand(5)
    print('X:', X)
    print('Y:', Y)

    Fxy = gram(f, X, Y)
    print('Fxy:', Fxy)
    # Fxy = gram(f, Y, X)
    # print('Fxy:', Fxy)

    # stat = infoNCE(f, X, Y)
    # print(stat)

    stat, p_value = permutation_test(f, X, Y)
    print(p_value)


if __name__=='__main__':
    main()
