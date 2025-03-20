import numpy as np
import torch
from torchvision.transforms import *


class NumpyToTensor:
    def __call__(self, ndarray):
        return torch.from_numpy(ndarray) #.float()
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

def identity(input):
    return input


def numpy_to_tensor(input: np.ndarray):
    return numpy_to_doubletensor(input)

def numpy_to_floattensor(input: np.ndarray):
    return torch.from_numpy(input).float()

def numpy_to_doubletensor(input: np.ndarray):
    return torch.from_numpy(input)
