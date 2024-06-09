import torch.nn as nn
from utils.utils import activation_registry


class ConvNet(nn.Module):
    def __init__(self,
                 channels: list[int],
                 kernel_size: list[int],
                 stride: list[int],
                 padding: list[int],
                 dilation: list[int],
                 activation: str = 'relu',
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 dropout: float = 0.,
                 last_nonlinear: bool = False,
                 init: str = 'default',
                 flatten: int = False,):  # TODO: replace with flatten
        super().__init__()
        n_layers = len(channels)
        if not (n_layers-1 == len(kernel_size) == len(stride) == len(padding) == len(dilation)):
            raise Exception(f"{self.__class__.__name__} expected the size of 'channels' to be one more than the size of 'kernel_size', 'stride', 'padding', and 'dilation'.")

        self.layers = nn.Sequential()
        for i in range(LASTLAYER := n_layers-1):            
            convBlock = ConvBlock(in_channels = channels[i],
                                  out_channels = channels[i+1],
                                  kernel_size = kernel_size[i],
                                  stride = stride[i],
                                  padding = padding[i],
                                  dilation = dilation[i],
                                  activation = activation  if (i<LASTLAYER-1 or last_nonlinear) else None,
                                  batch_norm = batch_norm  if (i<LASTLAYER-1 or last_nonlinear) else False,
                                  layer_norm = layer_norm  if (i<LASTLAYER-1 or last_nonlinear) else False,
                                  dropout = dropout        if (i<LASTLAYER-1 or last_nonlinear) else 0.,
                                  init = init)
            self.layers.append(convBlock)
        if flatten:
            self.layers.append(nn.Flatten(start_dim=-3))

    def forward(self, input):
        return self.layers(input)



class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 activation = 'relu',
                 batch_norm = False,
                 layer_norm = False,
                 dropout = 0.,
                 init = 'default'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_name = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.layer_norm = nn.LayerNorm() if layer_norm else None    # TODO: need to specify HxW of img
        self.activation = activation_registry(activation)
        self.dropout = nn.Dropout(dropout)
        self.init_weights(method=init)

    def init_weights(self, method = 'default'):
        if method == 'default': return
        if self.conv.bias != None:
            nn.init.constant_(self.conv.bias, val=0)
        if method == 'zeros':
            nn.init.constant_(self.conv.weight, val=0)
        elif method == 'normal':
            nn.init.normal_(self.conv.weight)
        elif method == 'xavier':
            nn.init.xavier_normal_(self.conv.weight)
        elif method == 'kaiming':
            nonlinearity = self.activation_name if self.activation_name != None else 'leaky_relu'
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity=nonlinearity)
        elif method == 'narrow_normal':
            nn.init.trunc_normal_(self.conv.weight, mean=0, std=1/self.in_channels, a=-2/self.in_channels, b=2/self.in_channels)

    def forward(self, input):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        elif self.layer_norm:
            x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

