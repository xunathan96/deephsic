import torch
import torch.nn as nn
import math
from .base import Kernel, BaseKernel
__all__ = ['Gaussian', 'GaussianJoint', 'WeightedGaussian', 'median_heuristic']

class Gaussian(BaseKernel):
    def __init__(self,
                 scale: float = 1.,
                 bandwidth: float = 1.,
                 flatten_input: bool = False,
                 trainable: bool = False):
        super().__init__()
        self.scale = scale
        self.flatten_input = flatten_input
        log_sigma = torch.empty(1)
        if trainable:
            nn.init.normal_(log_sigma, mean=math.log(bandwidth), std=0.25)  # 95% of samples in [μ-2*σ, μ+2*σ]
        else:
            nn.init.constant_(log_sigma, val=math.log(bandwidth))
        self.log_bandwidth = nn.Parameter(log_sigma, requires_grad=trainable)

    @property
    def bandwidth(self):
        return torch.exp(self.log_bandwidth)

    @bandwidth.setter
    def bandwidth(self, value: float):
        if value <= 0:
            raise Exception(f'bandwidth must be positive, but got {value}.')
        device = self.log_bandwidth.device
        self.log_bandwidth.data = torch.tensor([math.log(value)], device=device)

    def set_median_bandwidth(self, X: torch.Tensor, Y: torch.Tensor = None):
        if self.flatten_input:
            X = X.flatten(start_dim=1)
            if Y is not None:
                Y = Y.flatten(start_dim=1)
        median = median_heuristic(X, Y)
        self.bandwidth = median
        return median

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        if self.flatten_input:
            X = X.flatten(start_dim=1)
            Y = Y.flatten(start_dim=1)
        return self.gram(X, Y)

    def gram(self, X: torch.Tensor, Y: torch.Tensor):
        r"""compute the kernel gram matrix between samples X and Y
        X: (Nx, D) torch.Tensor
        Y: (Ny, D) torch.Tensor
        returns the gram matrix of size (Nx, Ny) with elements k(x_i, y_j)"""
        Dxy = pDist2(X, Y)    # (Nx, Ny)
        mahalanobis = -0.5*Dxy/(self.bandwidth**2)
        return self.scale * torch.exp(mahalanobis)
        # print('Dxy:', Dxy)
        # print('Dxy.max():', Dxy.max())  # for image x this max is 1730 !!
        # print('Dxy.min():', Dxy.min())  # 0
        # print('mahalanobis.max():', mahalanobis.max())
        # print('mahalanobis.min():', mahalanobis.min())
        # print('gram:', self.scale * torch.exp(mahalanobis))


class GaussianJoint(Gaussian):
    
    def gram(self, Z1: tuple[torch.Tensor], Z2: tuple[torch.Tensor]):
        # gaussian kernel on the joint sample Z1=(X1,Y1) and Z2=(X2,Y2)
        Z1 = torch.cat(Z1, dim=-1)  # (N, Dx+Dy)
        Z2 = torch.cat(Z2, dim=-1)  # (N, Dx+Dy)
        dist2 = pDist2(Z1, Z2)
        mahalanobis = -0.5*dist2/(self.bandwidth**2)
        return self.scale * torch.exp(mahalanobis)


class WeightedGaussian(BaseKernel):
    def __init__(self,
                 ndim: int = 1,
                 scale: float = 1.,
                 bandwidth: float = 1.,
                 trainable: bool = False):
        super().__init__()
        log_scale = torch.empty(ndim)
        log_sigma = torch.empty(ndim)
        if trainable:
            nn.init.normal_(log_scale, mean=math.log(scale), std=1)
            nn.init.normal_(log_sigma, mean=math.log(bandwidth), std=1)
        else:
            nn.init.constant_(log_scale, val=math.log(scale))
            nn.init.constant_(log_sigma, val=math.log(bandwidth))
        self.log_scale = nn.Parameter(log_scale, requires_grad=trainable)
        self.log_bandwidth = nn.Parameter(log_sigma, requires_grad=False)

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    @property
    def bandwidth(self):
        return torch.exp(self.log_bandwidth)

    def gram(self, X: torch.Tensor, Y: torch.Tensor):
        Dxy = (X.unsqueeze(dim=1) - Y)**2           # (M,N,D)
        mahalanobis = -0.5*Dxy/(self.bandwidth**2)  # (M,N,D)
        return torch.sum(self.scale * torch.exp(mahalanobis), dim=-1)   # (M,N)



class WeightedGaussian_2(Kernel):
    def __init__(self,
                 scale: tuple,
                 bandwidth: float = 1.,
                 device: torch.device = torch.device('cpu'),
                 trainable: bool = False):  # TODO...
        self.scale = torch.as_tensor(scale, dtype=torch.float, device=device)
        self.bandwidth = bandwidth

    def gram(self, X: torch.Tensor, Y: torch.Tensor):
        Dxy = (X.unsqueeze(dim=1) - Y)**2           # (M,N,D)
        mahalanobis = -0.5*Dxy/(self.bandwidth**2)  # (M,N,D)
        return torch.sum(self.scale * torch.exp(mahalanobis), dim=-1)   # (M,N)





def pDist2(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    r"""compute all paired (squared) distances between samples of X and Y
    X: (Nx, D) torch.Tensor
    Y: (Ny: D) torch.Tensor
    returns matrix of paired distances of size (Nx, Ny)"""
    xyT = X @ Y.mT                      # (Nx, Ny) pairwise inner products <x_i, y_j>
    x_norm2 = torch.sum(X**2, dim=-1)   # (Nx,)
    y_norm2 = torch.sum(Y**2, dim=-1)   # (Ny,)
    x_norm2 = x_norm2.unsqueeze(-1)     # (Nx, 1)
    Dxy = x_norm2 - 2*xyT + y_norm2     # (Nx, Ny) pairwise distances |x_i - y_j|^2
    Dxy[Dxy<0] = 0                      # TODO: clamp to stable values
    return Dxy



def median_heuristic_depreciated(*batches):
    r"""returns the median heuristic (√{D^2/2}) of all pairwise (squared) distances D^2
    between samples from each given batch of size (Ni,D)"""
    Z = torch.cat(batches, dim=0)
    Dzz = pDist2(Z,Z)
    upper_idx = torch.triu_indices(*Dzz.shape, offset=1)    # (2, N)
    dist2 = Dzz[upper_idx[0], upper_idx[1]]
    return torch.sqrt(dist2.median()/2).item()


def median_heuristic(X: torch.Tensor, Y: torch.Tensor = None):
    r"""returns the median heuristic (√{D^2/2}) of all pairwise (squared) distances D^2
    between samples from X to Y of size (Nx,D) and (Ny,D)."""
    Dxy = pDist2(X,Y) if Y is not None else pDist2(X, X)
    upper_idx = torch.triu_indices(*Dxy.shape, offset=1)    # (2, N)
    dist2 = Dxy[upper_idx[0], upper_idx[1]]
    return torch.sqrt(dist2.median()).item()
    return torch.sqrt(dist2.median()/2).item()



def main():
    kernel = Gaussian()

    X = torch.Tensor([
        [1],
        [2],
        [3],
    ])

    Kxx = kernel(X, X)
    print(Kxx)



if __name__=='__main__':
    main()

