import numpy as np
import torch
from .base import ToyDataset, ToyDataset_Depreciated
__all__ = ['Blob', 'BlobHD', 'Blob2D']


class Blob(ToyDataset_Depreciated):
    def __init__(self,
                 size: int,
                 dim: int,
                 n_modes: int,
                 means: list,
                 covs: list,
                 seed: int = None):
        self.dim = dim
        self.n_modes = n_modes
        self.means = means
        self.covs = covs
        super().__init__(size, seed)

    def sample(self, sample_shape):
        shape = sample_shape + (self.dim,)
        X = self.rng.standard_normal(shape) # (N,D)
        data = np.empty_like(X)
        mask = np.random.randint(self.n_modes, size=sample_shape)
        for i in range(self.n_modes):
            L = np.linalg.cholesky(self.covs[i])                # (D,D)
            data[mask==i] = self.means[i] + X[mask==i] @ L.T    # (N,D)
        return data


class Blob2D(Blob):
    def __init__(self,
                 size: int,
                 var: float = 0.03,
                 seed: int = None):
        dim = 2
        n_modes = 9
        means = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                means.append(np.array([i,j]))
        covs = []
        for i in range(1, 10):
            if i<5:
                delta = -0.02 - 0.002*(i-1)
            elif i>5:
                delta = 0.02 + 0.002*(i-6)
            else:
                delta = 0
            covs.append(
                np.array([
                    [var, delta],
                    [delta, var],
                ]))
        super().__init__(size, dim, n_modes, means, covs, seed)


class BlobHD(Blob):
    def __init__(self,
                 size: int,
                 dim: int = 2,
                 var: float = 1.,
                 seed: int = None):
        n_modes = 2
        means = [np.zeros((dim,)), np.zeros((dim,))]     # [np.zeros((dim,)), 0.5*np.ones((dim,))]
        cov1 = var*np.eye(dim)
        cov2 = var*np.eye(dim)
        cov1[0,1] = cov1[1,0] = 0.5
        cov2[0,1] = cov2[1,0] = -0.5
        covs = [cov1, cov2]
        super().__init__(size, dim, n_modes, means, covs, seed)
        """
        n_modes = 1
        means = [np.zeros((dim,))]
        cov1 = var*np.eye(dim)
        cov1[0,1] = cov1[1,0] = 0.5
        covs = [cov1]
        super().__init__(size, dim, n_modes, means, covs, seed)
        """



def main():
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    #dataset = Blob2D(size=1000, var=0.03)
    dataset = BlobHD(size=1000, dim=2, var=1)

    x = dataset.data
    plt.scatter(x[:,0], x[:,1], s=0.5)
    plt.axis('equal')
    plt.show()


if __name__=='__main__':
    main()


