import torch.nn as nn
from utils.utils import activation_registry


class FeedForward(nn.Module):
    def __init__(self,
                 features: list[int],
                 activation = 'relu',
                 batch_norm = False,
                 layer_norm = False,
                 dropout = 0.,
                 last_nonlinear = False,
                 init = 'default'):
        super().__init__()

        num_layers = len(features)
        self.layers = nn.Sequential()
        for i in range(LASTLAYER := num_layers-1):
            block = LinearBlock(features[i],
                                features[i+1],
                                activation  if (i<LASTLAYER-1 or last_nonlinear) else None,
                                batch_norm  if (i<LASTLAYER-1 or last_nonlinear) else False,
                                layer_norm  if (i<LASTLAYER-1 or last_nonlinear) else False,
                                dropout     if (i<LASTLAYER-1 or last_nonlinear) else 0.,
                                init)
            self.layers.append(block)

    def forward(self, input):
        return self.layers(input)



class LinearBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 activation = 'relu',
                 batch_norm = False,
                 layer_norm = False,
                 dropout = 0.,
                 init = 'default'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.linear = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.activation = activation_registry(activation)
        self.dropout = nn.Dropout(dropout)
        self.init_weights(method=init)

    def init_weights(self, method = 'default'):
        if method == 'default': return
        if self.linear.bias != None:
            nn.init.constant_(self.linear.bias, val=0)
        if method == 'zeros':
            nn.init.constant_(self.linear.weight, val=0)
        elif method == 'normal':
            nn.init.normal_(self.linear.weight)
        elif method == 'xavier':
            nn.init.xavier_normal_(self.linear.weight)
        elif method == 'kaiming':
            nonlinearity = self.activation_name if self.activation_name != None else 'leaky_relu'
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity=nonlinearity)
        elif method == 'narrow_normal':
            nn.init.trunc_normal_(self.linear.weight, mean=0, std=1/self.in_features, a=-2/self.in_features, b=2/self.in_features)

    def forward(self, input):
        x = self.linear(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        elif self.layer_norm:
            x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


