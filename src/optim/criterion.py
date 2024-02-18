# custom losses
import torch
import torch.nn as nn
from torch.nn import (BCELoss,
                      BCEWithLogitsLoss,
                      CrossEntropyLoss,
                      MSELoss)
import metrics
from kernel import Kernel


class MMDTestPower(nn.Module):
    def __init__(self,
                 reg: float = 1e-8):
        super().__init__()
        self.reg = reg

    def forward(self,
                k: Kernel,
                X: torch.Tensor,
                Y: torch.Tensor,) -> torch.Tensor:
        mmd2, var = metrics.mmd.mmd2(k, X, Y, statistic='u', onesampleU=True, compute_var=True)
        return -mmd2/torch.sqrt(var + self.reg) # loss = negative power


class HSICTestPower(nn.Module):
    def __init__(self,
                 reg: float = 1e-8):
        super().__init__()
        self.reg = reg

    def forward(self,
                k: Kernel,
                l: Kernel,
                X: torch.Tensor,
                Y: torch.Tensor,) -> torch.Tensor:
        hsic, var = metrics.hsic.hsic(k, l, X, Y, statistic='u', onesampleU=True, compute_var=True)
        return -hsic/torch.sqrt(var + self.reg) # loss = negative power








