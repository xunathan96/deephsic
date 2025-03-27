# custom losses
import math
import torch
import torch.nn as nn
from torch.nn import (BCELoss,
                      BCEWithLogitsLoss,
                      CrossEntropyLoss,
                      MSELoss)
import metrics
from kernel import Kernel
from distribution import gamma


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


class HSICTestPower_depreciated(nn.Module):
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


class HSICTestPower(nn.Module):
    def __init__(self,
                 with_threshold: bool = False,
                 significance: float = 0.05,
                 reg: float = 1e-8):
        super().__init__()
        self.with_threshold = with_threshold
        self.significance = significance
        self.reg = reg

    def forward(self,
                k: Kernel,
                l: Kernel,
                X: torch.Tensor,
                Y: torch.Tensor,) -> torch.Tensor:
        if self.with_threshold:
            return self.snr_w_thresh(k, l, X, Y)
        else:
            return self.snr_wo_thresh(k, l, X, Y)

    def snr_wo_thresh(self,
                      k: Kernel,
                      l: Kernel,
                      X: torch.Tensor,
                      Y: torch.Tensor,) -> torch.Tensor:
        hsic, var = metrics.hsic.hsic(k, l, X, Y, statistic='u', compute_var=True)
        return - hsic / torch.sqrt(var + self.reg)

    def snr_w_thresh(self,
                     k: Kernel,
                     l: Kernel,
                     X: torch.Tensor,
                     Y: torch.Tensor,) -> torch.Tensor:
        m = X.shape[0]
        Kxx = k(X, X)
        Lyy = l(Y, Y)
        hsic, var = metrics.hsic.hsic_fast(Kxx, Lyy, statistic='v', compute_var=True)
        # asymptotic threshold based on (differentiable) gamma approx.
        e0 = metrics.hsic.null_mean(Kxx, Lyy)
        v0 = metrics.hsic.null_var(Kxx, Lyy)
        shape = torch.atleast_1d(e0**2 / v0)
        scale = torch.atleast_1d(v0 / e0)
        r = gamma.icdf(1-self.significance, shape, scale)
        # signal & thrsehold-to-noise ratios
        std = torch.sqrt(var + self.reg)
        snr = hsic / std
        tnr = r / std
        return - ( math.sqrt(m) * snr - tnr / math.sqrt(m) )




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


class MITestPower(nn.Module):
    def __init__(self,
                 normalize: bool = False,
                 reg: float = 1e-8):
        super().__init__()
        self.reg = reg
        self.normalize = normalize
    
    def forward(self,
                f: nn.Module,
                X: torch.Tensor,
                Y: torch.Tensor,) -> torch.Tensor:
        pscore, var = metrics.mi.pairscore(f, X, Y)
        if self.normalize:
            return - pscore / torch.sqrt(var + self.reg)
        else:
            return - pscore


# class MITestPower_old(nn.Module):
#     def __init__(self,
#                  normalize: bool = False,
#                  reg: float = 1e-8):
#         super().__init__()
#         self.reg = reg
#         self.normalize = normalize
    
#     def forward(self,
#                 f: nn.Module,
#                 X: torch.Tensor,
#                 Y: torch.Tensor,) -> torch.Tensor:
#         T1, var = metrics.mi.T(f, X, Y)
#         if not self.normalize:
#             # maximize T / var
#             return - T1 / torch.sqrt(var + self.reg)
#         else:
#             # maximize (T - T0) / var
#             Fxy = metrics.mi.gram(f, X, Y)  # (n,n)
#             # T0 = torch.mean(Fxy)
#             n = Fxy.shape[-1]
#             T0 = (torch.sum(Fxy) - torch.trace(Fxy)) / (n*(n-1))
#             return - (T1 - T0) / torch.sqrt(var + self.reg)

