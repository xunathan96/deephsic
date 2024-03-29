import torch
import torch.nn as nn
from .base import Identity


class Basic(nn.Module):

    expansion = 1
    
    def __init__(self, in_channels, inter_channels, stride=1):
        super().__init__()
        out_channels = self.expansion * inter_channels
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=stride, padding=1, bias=False) # bias term accounted for in batch norm
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)                 # TODO: try layernorm instead?
        self.act = nn.ELU()                                     # TODO: make this argument?
        self.projection = projected_downsample(in_channels, out_channels, stride)

    def forward(self, input):
        x = self.act(self.bn1(self.conv1(input)))   # (3x3, inter_channels           / stride)
        x = self.bn2(self.conv2(x))                 # (3x3, inter_channels*expansion / 1)
        x = self.act(x + self.projection(input))
        return x


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, inter_channels, stride=1):
        super().__init__()
        out_channels = self.expansion * inter_channels
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU()
        self.projection = projected_downsample(in_channels, out_channels, stride)

    def forward(self, input):
        x = self.act(self.bn1(self.conv1(input)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.act(x + self.projection(input))
        return x


class ResNet(nn.Module):

    block_factory = {'basic': Basic, 'bottleneck': Bottleneck,}

    def __init__(self,
                 inter_channels: list[int],
                 n_blocks: list[int],
                 in_channels: int = 3,
                 out_features: int = 1000,
                 block_type: str = 'basic',
                 ):
        r"""ResNet architecture.
        inter_channels: list of number of (intermediate) channels for each layer.
        n_blocks:       list of number of residual blocks for each layer.
        in_channels:    number of input channels.
        out_features:   number of output features.
        block_type:     either 'basic' or 'bottleneck'."""
        super().__init__()
        self.inter_channels = inter_channels
        self.n_blocks = n_blocks
        self.block_type = block_type
        self.out_features = out_features
        expansion = ResNet.block_factory[block_type].expansion

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ELU() # TODO: make argument
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, self.inter_channels[0], self.n_blocks[0], stride=1)
        self.layer2 = self._make_layer(expansion * self.inter_channels[0], self.inter_channels[1], self.n_blocks[1], stride=2)
        self.layer3 = self._make_layer(expansion * self.inter_channels[1], self.inter_channels[2], self.n_blocks[2], stride=2)
        self.layer4 = self._make_layer(expansion * self.inter_channels[2], self.inter_channels[3], self.n_blocks[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(expansion * self.inter_channels[3], out_features)

    def forward(self, input):
        x = self.act(self.bn1(self.conv1(input)))   # (N, 64, 112, 112)
        x = self.max_pool(x)                        # (N, 64, 56, 56)
        x = self.layer1(x)                          # (N, expansion*inter_channel[0], 56, 56)
        x = self.layer2(x)                          # (N, expansion*inter_channel[1], 28, 28)
        x = self.layer3(x)                          # (N, expansion*inter_channel[2], 14, 14)
        x = self.layer4(x)                          # (N, expansion*inter_channel[3], 7, 7)
        x = self.avg_pool(x)                        # (N, expansion*inter_channel[3], 1, 1)
        x = self.fc1(x.flatten(start_dim=-3))       # (N, out_features)
        return x

    def _make_layer(self, in_channels, inter_channels, n_blocks, stride):
        r"""builds a sequence of residual blocks.
        in_channels:    number of input channels.
        inter_channels: number of channels in each residual block input.
        n_blocks:       number of residual blocks in the layer.
        stride:         stride used in the first residual block.
        """
        Residual = ResNet.block_factory[self.block_type]
        out_channels = Residual.expansion * inter_channels
        
        blocks = []
        blocks.append(Residual(in_channels, inter_channels, stride=stride))
        for i in range(n_blocks-1):
            blocks.append(Residual(out_channels, inter_channels, stride=1))

        return nn.Sequential(*blocks)


# ==============================
#       RESNET VARIANTS
# ==============================

class ResNet18(ResNet):
    def __init__(self): super().__init__(**model_parameters['resnet18'])
class ResNet34(ResNet):
    def __init__(self): super().__init__(**model_parameters['resnet34'])
class ResNet50(ResNet):
    def __init__(self): super().__init__(**model_parameters['resnet50'])
class ResNet101(ResNet):
    def __init__(self): super().__init__(**model_parameters['resnet101'])
class ResNet152(ResNet):
    def __init__(self): super().__init__(**model_parameters['resnet152'])

model_parameters = dict()
model_parameters['resnet18'] = {
    'inter_channels': [64, 128, 256, 512],
    'n_blocks': [2, 2, 2, 2],
    'in_channels': 3,
    'out_features': 1000,
    'block_type': 'basic',}
model_parameters['resnet34'] = {
    'inter_channels': [64, 128, 256, 512],
    'n_blocks': [3, 4, 6, 3],
    'in_channels': 3,
    'out_features': 1000,
    'block_type': 'basic',}
model_parameters['resnet50'] = {
    'inter_channels': [64, 128, 256, 512],
    'n_blocks': [3, 4, 6, 3],
    'in_channels': 3,
    'out_features': 1000,
    'block_type': 'bottleneck',}
model_parameters['resnet101'] = {
    'inter_channels': [64, 128, 256, 512],
    'n_blocks': [3, 4, 23, 3],
    'in_channels': 3,
    'out_features': 1000,
    'block_type': 'bottleneck',}
model_parameters['resnet152'] = {
    'inter_channels': [64, 128, 256, 512],
    'n_blocks': [3, 8, 36, 3],
    'in_channels': 3,
    'out_features': 1000,
    'block_type': 'bottleneck',}


# ==============================
#       HELPER FUNCTIONS
# ==============================

def projected_downsample(in_channels: int,
                         out_channels: int,
                         stride: int,
                         ) -> nn.Module:
    if stride == 1 and in_channels == out_channels:
        projection = Identity()
    else:
        projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    return projection







# ==============================
#           TESTING
# ==============================

def main():
    device = torch.device('cpu')
    x = torch.rand((128, 3, 224, 224)).to(device)
    resnet = ResNet50().to(device)
    y = resnet(x)
    print(y.shape)

if __name__=='__main__':
    main()
