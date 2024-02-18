import torch.nn as nn
from utils.utils import activation_registry

class CNN(nn.Module):
    
    def __init__(self,
                 channels: list[int],
                 kernel_size: list[int],
                 stride: list[int],
                 padding: list[int],
                 dilation: list[int],
                 activation='ReLU',
                 batch_norm=False,
                 flatten: int = False,
                 dropout=0.):
        super().__init__()

        n_layers = len(channels)
        if not (n_layers-1 == len(kernel_size) == len(stride) == len(padding) == len(dilation)):
            raise Exception(f"{self.__class__.__name__} expected the size of 'channels' to be one more than the size of 'kernel_size', 'stride', 'padding', and 'dilation'.")

        self.net = nn.Sequential()
        for i in range(LASTLAYER := n_layers-1):            
            convBlock = ConvBlock(in_channels=channels[i],
                                  out_channels=channels[i+1],
                                  kernel_size=kernel_size[i],
                                  stride=stride[i],
                                  padding=padding[i],
                                  dilation=dilation[i],
                                  activation=activation  if i<LASTLAYER-1 else None,
                                  batch_norm=batch_norm  if i<LASTLAYER-1 else False,
                                  dropout=dropout        if i<LASTLAYER-1 else 0.,)
            self.net.append(convBlock)
        if flatten:
            self.net.append(nn.Flatten(start_dim=-3))

    def forward(self, input):
        return self.net(input)



class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 activation='ReLU',
                 batch_norm=False,
                 layer_norm=False,
                 dropout=0.):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.layer_norm = nn.LayerNorm() if layer_norm else None
        self.activation = activation_registry(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x



class RegressionHead(nn.Module):
    ...


class ClassificationHead(nn.Module):
    ...




