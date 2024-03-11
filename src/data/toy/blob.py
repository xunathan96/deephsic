import numpy as np
import torch
from .base import ToyDataset
__all__ = ['Blob', 'Blob2ST']


class Blob(ToyDataset):
    def __init__(self,
                 n_samples: int,
                 means: list[np.ndarray],
                 covs: list[np.ndarray],
                 weights: list[float]):

        self.n_mixtures = len(means)
        if not (self.n_mixtures == len(covs) == len(weights)):
            raise Exception(f"Error: the number of mixtures specified must be consistent with the size of means, covs, and weights.")

        self.dim = means[0].shape[-1]
        for mean,cov in zip(means, covs):
            if mean.shape != (self.dim,) or cov.shape != (self.dim,self.dim):
                raise Exception(f"Error: the means and covs must have consistent dimension size.")

        self.means = means
        self.covs = covs
        self.chols = [np.linalg.cholesky(cov) for cov in covs]
        self.weights = weights
        super().__init__(n_samples)

    def __len__(self):
        return len(self.data[0])    # otherwise max batch size is 2

    def __getitem__(self, idx):
        return tuple(torch.from_numpy(x[idx]).float() for x in self.data)

    def sample(self, shape: tuple):
        size = shape + (self.dim,)
        data = list()
        for i in range(self.n_mixtures):
            eps = np.random.normal(size=size)
            gaussian = self.means[i] + eps @ self.chols[i].T
            data.append(gaussian)   # (N, D)
        return data

    @staticmethod
    def collate(batch):
        ...


class Blob2ST(Blob):
    def __init__(self,
                 size: int,
                 dim: int,):
        weights = [1., 1.]
        means = [np.zeros(dim), np.zeros(dim)]
        cov1 = np.eye(dim)
        cov2 = np.eye(dim)
        cov1[0,1] = cov1[1,0] = 0.55 #0.5
        cov2[0,1] = cov2[1,0] = 0.45 #-0.5
        covs = [cov1, cov2]
        super().__init__(size, means, covs, weights)





def main():
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    blob = Blob2ST(size=1000, dim=4)
    dataloader = DataLoader(blob, batch_size=1000)

    diter = iter(dataloader)
    x,y = next(diter)
    print(x.shape)
    print(y.shape)

    # x, y = blob.data
    plt.scatter(x[:,0], x[:,1], s=1)
    plt.scatter(y[:,0], y[:,1], s=1)
    plt.show()


if __name__=='__main__':
    main()

