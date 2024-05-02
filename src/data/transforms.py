import torch
from torchvision.transforms import *


class NumpyToTensor:
    def __call__(self, ndarray):
        return torch.from_numpy(ndarray).float()
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
