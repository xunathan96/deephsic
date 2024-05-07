import torch
import torch.nn as nn
from tqdm import tqdm


def gram(f: nn.Module,
         X: torch.Tensor,
         Y: torch.Tensor) -> torch.Tensor:
    r"""Computes the square gram matrix f(xi,yj) given batches X=(x1,...,xn) and Y=(y1,...,yn)."""
    if (n:=X.size(0)) != Y.size(0):
        raise Exception(f"Error: expected batches X and Y to have equal number of samples.")
    device = X.device
    fxy = torch.empty((n,n), device=device)
    idx = torch.arange(n, device=device)
    idy = torch.arange(n, device=device)
    for i in range(n):
        shift_idy = idy - i
        Y_shift = Y[shift_idy]  # shift Y to right
        fxy[idx, shift_idy] = f(X, Y_shift)
    return fxy


def infoNCE(f: nn.Module,
            X: torch.Tensor,
            Y: torch.Tensor,):
    r"""Computes the InfoNCE metric for samples (Xi, Yi) ~ Pxy."""
    fxy = gram(f, X, Y)
    return infoNCE_fast(fxy)


def infoNCE_fast(fxy: torch.Tensor):
    n = fxy.size(0)
    trace = torch.einsum('ii', fxy)
    return trace/n - torch.logsumexp(fxy, dim=-1).mean() + torch.log(n)




def permutation_test(f: nn.Module,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     n_permutations: int = 500):
    r"""perform a permutation test to compute the test statistic and p-value.
    f: Pytorch model with scalar outputs
    X: (N,*,Dx) samples from Pxy
    Y: (N,*,Dy) samples from Pxy
    statistic: 'accuracy' for c2st-s or 'logit' for c2st-l
    returns the test statistic and its p-value under the null hypothesis
    """
    n = X.shape[0]
    device = X.device
    # C2ST test statistic is (soft) accuracy between samples Z=(X,Y) from null and alternate distributions
    shuffle_idx = torch.randperm(n, device=device)
    Y_null = Y[shuffle_idx]
    Z_null = (X,Y_null)
    Z_alt = (X,Y)
    # get test samples
    X_test, Y_test = catzip(Z_null, Z_alt, dim=0)   # (2N,*,Dx) and (2N,*,Dy)
    # get test labels
    t_null = torch.zeros(2*n, device=device)    # (2N,)
    t_alt = torch.cat((
        torch.zeros(n, device=device),
        torch.ones(n, device=device)))

    # compute test statistic (accuracy)
    logits = classifier(X_test, Y_test).squeeze(-1) # (2N,)
    if statistic == 'accuracy':
        acc = accuracy_with_logits(logits, t_alt)
    elif statistic == 'logit':
        acc = soft_accuracy_with_logits(logits, t_alt)

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples in Y for null hypothesis (i.e. Px*Py)
        shuffle_idx = torch.randperm(n, device=device)
        Y_shuffled = Y[shuffle_idx]
        Y_test = torch.cat((Y_null, Y_shuffled), dim=0) # (2N,*,Dy)
        # compute test statistics under the null distribution
        logits = classifier(X_test, Y_test).squeeze(-1)
        if statistic == 'accuracy':
            acc_null = accuracy_with_logits(logits, t_alt)
        elif statistic == 'logit':
            acc_null = soft_accuracy_with_logits(logits, t_alt)
        stats.append(acc_null.item())
        if acc_null >= acc: # NOTE: use >= to account for equal accuracy statistics
            count += 1

    p_value = count/n_permutations
    return acc.item(), p_value




def main():
    class DummyNet(nn.Module):
        def forward(self, X, Y):
            return X + 2*Y

    f = DummyNet()
    X = torch.arange(5)+0.0
    Y = torch.arange(5)+100.0
    print('X:', X)
    print('Y:', Y)

    g = gram(f, X, Y)
    print(g)

    g = gram(f, Y, X)
    print(g)


if __name__=='__main__':
    main()
