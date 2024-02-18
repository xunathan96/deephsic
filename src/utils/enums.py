from enum import Enum
import torch.nn as nn

class YamlNodeType(Enum):
    SCALAR = 0
    SEQUENCE = 1
    MAPPING = 2

def ActivationFactoryY(activation, *args, **kwds):
    return {
        None: nn.Identity,
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'Tanh': nn.Tanh,
        'GLU': nn.GLU,
        'ELU': nn.ELU,
        'Sigmoid': nn.Sigmoid,
        'Softmax': nn.Softmax,
    }[activation](*args, **kwds)

