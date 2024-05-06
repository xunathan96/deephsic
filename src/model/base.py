import torch
import torch.nn as nn
__all__ = ['Identity', 'Neck', 'activation_registry']

class Identity(nn.Module):
    def forward(self, input):
        return input

class Sequential(nn.Sequential):
    def __init__(self, modules: list[nn.Module]):
        super().__init__(*modules)

class Neck(nn.Module):
    def __init__(self,
                 backbones: nn.ModuleList,
                 heads: nn.ModuleList,
                 connection: str = 'cat',
                 squeeze_single_head: bool = True,
                 ) -> list[torch.Tensor]:
        super().__init__()
        self.connection = connection
        self.backbones = backbones
        self.heads = heads
        self.squeeze_single_head = squeeze_single_head
        self.n_inputs = len(backbones)
        self.n_outputs = len(heads)

    def forward(self, *inputs):
        if len(inputs)!=self.n_inputs:
            raise Exception(f'Error: expected {self.n_inputs} but got only {len(inputs)}')

        encodings = list()
        for input, backbone in zip(inputs, self.backbones):
            encodings.append(backbone(input))   # (*, d_i)

        if self.connection == 'cat':
            latent = torch.cat(encodings, dim=-1)   # (*, d_1+...d_n)
        elif self.connection == 'add':
            latent = torch.stack(encodings, dim=-1).sum(dim=-1)  # (*, d)
        else:
            raise NotImplementedError()

        outputs = list()
        for head in self.heads:
            outputs.append(head(latent))
        
        if self.squeeze_single_head and self.n_outputs==1:
            return outputs[0]
        return outputs



class ModelFactory:
    ...



def activation_registry(activation, *args, **kwds) -> nn.Module:
    return {
        None: nn.Identity,
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'Tanh': nn.Tanh,
        'GLU': nn.GLU,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
        'Sigmoid': nn.Sigmoid,
        'Softmax': nn.Softmax,
    }[activation](*args, **kwds)



# build model based on given configuration