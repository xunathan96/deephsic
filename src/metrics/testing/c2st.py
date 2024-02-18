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
    preds = torch.zeros_like(probs)
    preds[probs>=0.5] = 1
    return accuracy(preds, t)


def permutation_test(classifier: nn.Module,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     n_permutations: int = 500):
    r"""perform a permutation test to compute the accuracy and p-value of the test statistic.
    classifier: Pytorch model with output as unnormalized logits
    X: (N,D) samples from Px*Py
    Y: (N,D) samples from Px*Py
    returns the test statistic (accuracy) and its p-value under the null hypothesis
    """
    n = X.shape[0]
    device = X.device
    t_alt = torch.ones(2*n, device=device)

    Z_alt = torch.cat((X,Y), dim=-1)    # (N,2D)
    logits = classifier(Z_alt)
    acc = accuracy_with_logits(logits, t_alt)   # C2ST test statistic is accuracy given all labels are 1 (alternate samples)

    count = 0
    stats = []
    for i in tqdm(range(n_permutations),
                  bar_format="running permutation test... |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True,
                  leave=False):
        # permute samples in Y for null hypothesis (i.e. Pxy = Px*Py)
        shuffle_idx = torch.randperm(n, device=device)
        Y_shuffled = Y[shuffle_idx]
        # compute test statistics under the null distribution
        Z_null = torch.cat((X,Y_shuffled), dim=-1)  # (N,2D)
        logits = classifier(Z_null)
        acc_null = accuracy_with_logits(logits, t_alt)
        stats.append(acc_null.item())
        if acc_null > acc:
            count += 1

    p_value = count/n_permutations
    return acc.item(), p_value  # return stats?
