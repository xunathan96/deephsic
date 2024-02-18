import torch.nn as nn
from utils.utils import activation_registry


class MLP(nn.Module):
    def __init__(self,
                 features: list[int],
                 activation='ReLU',
                 batch_norm=False,
                 layer_norm=False,
                 dropout=0.):
        super().__init__()
        n_layers = len(features)
        self.net = nn.Sequential()
        for i in range(LASTLAYER := n_layers-1):
            linearBlock = LinearBlock(in_features=features[i],
                                      out_features=features[i+1],
                                      activation=activation  if i<LASTLAYER-1 else None,
                                      batch_norm=batch_norm  if i<LASTLAYER-1 else False,
                                      layer_norm=layer_norm  if i<LASTLAYER-1 else False,
                                      dropout=dropout        if i<LASTLAYER-1 else 0.)
            self.net.append(linearBlock)

    def forward(self, input):
        return self.net(input)




class LinearBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 activation='ReLU',
                 batch_norm=False,
                 layer_norm=False,
                 dropout=0.):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.activation = activation_registry(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        x = self.linear(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        elif self.layer_norm:
            x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


