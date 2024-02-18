import numpy as np
from .base import ToyDataset
__all__ = ['Gaussian2D']


class Gaussian2D(ToyDataset):
    def __init__(self,
                 size: int,
                 mean=(0,0),
                 var=(1,1),
                 cor=0,
                 seed: int = None):
        if not -1<cor<1:
            raise Exception(f'Expected correlation to be in (-1, 1) but got {cor}.')
        self.mean = np.array(mean)
        self.var = np.array(var)
        self.cor = cor
        std = (np.sqrt(var[0]), np.sqrt(var[1]))
        self.cov = np.array([
            [var[0], cor*std[0]*std[1]],
            [cor*std[0]*std[1], var[1]]
        ])
        super().__init__(size, seed)

    def sample(self, sample_shape):
        size = sample_shape + (2,)
        X = self.rng.standard_normal(size)  # (N,2)
        L = np.linalg.cholesky(self.cov)    # (2,2)
        data = self.mean + X @ L.T          # (N,2)
        return data

class IsotropicGaussian2D(Gaussian2D):
    def __init__(self,
                 size: int,
                 mean=(0, 0),
                 var=(1, 1),
                 seed: int = None):
        super().__init__(size, mean, var, cor=0, seed=seed)




def main():

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    dataset = Gaussian2D(size=10000, var=(10, 0.1), cor=0.5)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

    x = dataset.data
    plt.scatter(x[:,0], x[:,1], s=0.5)
    plt.axis('equal')
    plt.show()


if __name__=='__main__':
    main()
