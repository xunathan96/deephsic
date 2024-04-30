import torch
import torch.nn as nn
from tqdm import tqdm


def accuracy(pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""compute the classification accuracy based on the output predictions and labels.
    pred:   (N,) torch.Tensor with values {0.,1.}, where 0==null sample, 1==alt sample
    t:      (N,) torch,Tensor with values {0.,1.}
    returns the scalar accuracy."""
    return (pred==t).float().mean()

def accuracy_with_logits(logits: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # compute the accuracy given the logits (N,) and labels (N,)
    probs = torch.sigmoid(logits)
    return accuracy((probs>=0.5).float(), t)

def soft_accuracy_with_logits(logits: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # compute the soft accuracy based on logits (N,) and labels (N,)
    # logits = z_0 - z_1
    # logits > 0 --> P(y=1|x)>0.5
    # logits < 0 --> P(y=1|x)<0.5
    return logits[t==1].mean() - logits[t==0].mean()



def permutation_test(classifier: nn.Module,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     statistic: str = 'logit',
                     n_permutations: int = 500):
    r"""perform a permutation test to compute the test statistic and p-value.
    classifier: Pytorch model with output as unnormalized logits
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
            acc_null = accuracy_with_logits(logits, t_null)
        elif statistic == 'logit':
            acc_null = soft_accuracy_with_logits(logits, t_null)
        stats.append(acc_null.item())
        if acc_null > acc:
            count += 1

    p_value = count/n_permutations
    return acc.item(), p_value


def permutation_test_depreciated(classifier: nn.Module,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     n_permutations: int = 500):
    r"""perform a permutation test to compute the accuracy and p-value of the test statistic.
    classifier: Pytorch model with output as unnormalized logits
    X: (N,*,Dx) samples from Pxy
    Y: (N,*,Dy) samples from Pxy
    returns the test statistic (accuracy) and its p-value under the null hypothesis
    """
    n = X.shape[0]
    device = X.device
    # C2ST test statistic is accuracy given all labels are 1 (alternate samples)
    logits = classifier(X,Y).squeeze(-1)    # (N,)
    t_alt = torch.ones(n, device=device)
    acc = accuracy_with_logits(logits, t_alt)

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples in Y for null hypothesis (i.e. Px*Py)
        shuffle_idx = torch.randperm(n, device=device)
        Y_shuffled = Y[shuffle_idx]
        # compute test statistics under the null distribution
        logits = classifier(X, Y_shuffled).squeeze(-1)
        acc_null = accuracy_with_logits(logits, t_alt)
        stats.append(acc_null.item())
        if acc_null > acc:
            count += 1

    p_value = count/n_permutations
    return acc.item(), p_value



# ==============================
#       HELPER FUNCTIONS
# ==============================

def catzip(*iters, dim=0):
    # zips the given iterators and then applies torch.cat on each zipped tuple
    return (torch.cat(tensors, dim=dim) for tensors in zip(*iters))




