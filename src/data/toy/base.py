import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class ToyDataset(Dataset):
    
    def __init__(self, n_samples: int):
        super().__init__()
        self.data = self.sample((n_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()

    @abstractmethod
    def sample(self, shape):
        ...



class ToyDataset_Depreciated(Dataset, ABC):

    def __init__(self,
                 size: int,
                 seed: int = None):
        self.size = size
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.data = self.sample((self.size,))

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()

    def resample(self):
        self.data = self.sample((self.size,))
        return self.data

    @abstractmethod
    def sample(self, sample_shape):
        ...
