# custom losses
import torch
import torch.nn as nn
from torch.nn import (BCELoss,
                      BCEWithLogitsLoss,
                      CrossEntropyLoss,
                      MSELoss)
import metrics
from kernel import Kernel


class HSIC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                k: Kernel,
                l: Kernel,
                X: torch.Tensor,
                Y: torch.Tensor,) -> torch.Tensor:
        hsic, var = metrics.hsic.hsic(k, l, X, Y, statistic='u', onesampleU=True, compute_var=False)
        return -hsic


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


class MutualInfoLowerBound(nn.Module):
    def __init__(self,
                 bound: str):
        super().__init__()
        self.metric = {
            'donsker_varadhan': NotImplementedError(),
            'mine': NotImplementedError(),
            'tuba': NotImplementedError(),
            'info_nce': metrics.infonce.infoNCE,
            'nwj': metrics.nwj.nwj,
        }[bound]

    def forward(self,
                f: nn.Module,
                X: torch.Tensor,
                Y: torch.Tensor,) -> torch.Tensor:
        return -self.metric(f, X, Y)





