import numpy as np
import torch
from .base import ToyDataset
__all__ = ['Sinusoid']


class Sinusoid(ToyDataset):

    def __init__(self,
                 size: int,
                 frequency: float,
                 dim: int = 1,
                 split: str = 'train',
                 train_val_test_split: str = '7:1:2'):
        self.frequency = frequency
        self.dim = dim
        super().__init__(size)

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

    def sample(self, shape: tuple):
        samples = list()
        n = 0
        n_samples = shape[0]

        while n < n_samples:
            samples_accept = self.rejection_sampling(n_samples)
            samples.append(samples_accept)
            n += len(samples_accept)

        samples = np.concatenate(samples, axis=0)[:n_samples]
        return samples


    def rejection_sampling(self, size: int):
        l = self.frequency
        samples_proposal = np.pi * (2*np.random.rand(size, 2) - 1)
        density_unnorm = 1 + np.sin(l * samples_proposal[:,0]) * np.sin(l * samples_proposal[:,1])
        keep_prob = density_unnorm / 2  # p(x)/{Mq(x)}
        alpha = np.random.rand(size)
        if self.dim > 1:
            samples_unif = np.pi * (2*np.random.rand(size, 2*(self.dim-1)) - 1)
            samples_proposal = np.concatenate((samples_proposal, samples_unif), axis=-1)    # (N, 2d)
            # move dependent axes to first and last axes
            reorder_idx = np.arange(2*self.dim)
            reorder_idx[1] = 2*self.dim - 1
            reorder_idx[-1] = 1
            samples_proposal = samples_proposal[:, reorder_idx]
        return samples_proposal[alpha < keep_prob]



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

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    dataset = Sinusoid(size=1000000,
                       frequency=4,
                       dim=1,
                       split='train',
                       train_val_test_split='5000:0:')
    print('size:', len(dataset))

    dataloader = DataLoader(dataset, batch_size=10000, shuffle=True)
    batch = next(iter(dataloader))
    X,Y = batch
    print(X.shape)
    print(Y.shape)

    plt.scatter(X[:,0], Y[:,-1], s=2)
    plt.axis('square')

    plt.xlim(left=-3.2, right=3.2)
    plt.ylim(top=3.2, bottom=-3.2)
    plt.xticks([])
    plt.yticks([])

    plt.show()
    # plt.savefig(f"density-1.pdf", format="pdf", bbox_inches="tight")

    # samples = dataset.data
    # plt.scatter(samples[:,0], samples[:,5], s=2)
    # plt.axis('equal')
    # plt.show()



if __name__ == '__main__':
    main()




