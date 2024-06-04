import numpy as np
import torch
from .base import ToyDataset
__all__ = ['GaussianMixture', 'HDGM']


class GaussianMixture(ToyDataset):
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


    def sample(self, shape: tuple, deterministic: bool = True):
        if deterministic: np.random.seed(2024)
        size = shape + (self.dim,)
        eps = np.random.normal(size=size)   # (N,D)
        data = np.empty_like(eps)
        mix_id = np.random.randint(self.n_mixtures, size=shape)  # TODO: categorical prob based on weights
        for i in range(self.n_mixtures):
            data[mix_id==i] = self.means[i] + eps[mix_id==i] @ self.chols[i].T
        return data


class HDGM(GaussianMixture):
    def __init__(self,
                 size: int,
                 dim: int,
                 split: str = None,
                 train_val_test_split: str = '7:1:2',):
        weights = [1., 1.]
        means = [np.zeros(dim), np.zeros(dim)]
        cov1 = np.eye(dim)
        cov2 = np.eye(dim)
        cov1[0,3] = cov1[3,0] = 0.5
        cov2[0,3] = cov2[3,0] = -0.5
        covs = [cov1, cov2]
        super().__init__(size, means, covs, weights)

        # compute train-val-test splits
        splits = [int(m) if m.isdigit() else None for m in train_val_test_split.split(':')]
        # one split is inferred
        if splits.count(None) > 1:
            raise Exception(f"We can only infer a maximum of one split.")
        splits = [size-sum(filter(None, splits)) if m is None else m for m in splits]

        TRAIN_SPLIT = splits[0]/sum(splits)
        VAL_SPLIT = splits[1]/sum(splits)
        TEST_SPLIT = splits[2]/sum(splits)

        # data splits
        split_idx = np.cumsum([int(split*size) for split in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)])
        if split == 'train':
            self.data = self.data[:split_idx[0]]
        elif split == 'val':
            self.data = self.data[split_idx[0]:split_idx[1]]
        elif split == 'test':
            self.data = self.data[split_idx[1]:split_idx[2]]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        xy = torch.from_numpy(self.data[idx]).float()
        return marginals(xy)



def marginals(joint: np.ndarray | torch.Tensor):
    # split joint samples into marginals X,Y
    dim = joint.shape[-1]
    if isinstance(joint, np.ndarray):
        mask = np.ones(dim, dtype=bool)
    elif isinstance(joint, torch.Tensor):
        mask = torch.ones(dim, dtype=torch.bool)
    mask[1::2] = False
    if joint.ndim == 1:
        return joint[mask], joint[~mask]
    elif joint.ndim == 2:
        return joint[:,mask], joint[:,~mask]
    else:
        raise NotImplementedError()



def main():
    hdgm = HDGM(size=10000,
                    dim=4,
                    split='test',
                    train_val_test_split='7:1:2')
    print(len(hdgm))


if __name__=='__main__':
    main()
