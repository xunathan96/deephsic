import torch
import torch.nn as nn
from .attention import SelfAttention
from .base import activation_registry
__all__=['Transformer', 'TransformerEncoder', 'TransformerBlock', 'PositionalEncoding']


class TransformerEncoder(nn.Module):
    def __init__(self,
                 seq_len,
                 embed_dim,
                 num_heads = 1,
                 expansion_factor = 4,
                 num_layers = 4,
                 activation = 'ReLU',
                 drop_attn = 0,
                 drop_ff = 0,):
        super().__init__()

        layers = []
        for i in range(num_layers):
            block = TransformerBlock(embed_dim,
                                     num_heads,
                                     expansion_factor,
                                     activation,
                                     drop_attn,
                                     drop_ff,)

            layers.append(block)
        self.blocks = nn.Sequential(*layers)
        self.pos_encoder = PositionalEncoding(seq_len, embed_dim)

    def forward(self, input):
        return self.blocks(self.pos_encoder(input))


class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads = 1,
                 expansion_factor = 4,
                 activation = 'ReLU',
                 drop_attn = 0,
                 drop_ff = 0,
                 drop_score = 0,
                 drop_proj = 0,):
        super().__init__()

        self.attn = SelfAttention(input_dim = embed_dim,
                                  embed_dim = embed_dim,
                                  output_dim = embed_dim,
                                  num_heads = num_heads,
                                  drop_score = drop_score,
                                  drop_proj = drop_proj,)

        self.ff = nn.Sequential(nn.Linear(embed_dim, expansion_factor*embed_dim),
                                activation_registry(activation),
                                nn.Linear(expansion_factor*embed_dim, embed_dim),)

        self.norm_attn = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.dropout_attn = nn.Dropout(drop_attn)
        self.dropout_ff = nn.Dropout(drop_ff)

    def forward(self, input):
        # (N, T, D) -> (N, T, D)
        x = self.attn(input)
        x = self.dropout_attn(self.norm_attn(x + input))
        h = self.ff(x)
        h = self.dropout_ff(self.norm_ff(h + x))
        return h


class PositionalEncoding(nn.Module):
    def __init__(self,
                 seq_len,
                 embed_dim):
        super().__init__()

        self.encoding = torch.zeros((seq_len, embed_dim))
        pos = 1 + torch.arange(seq_len).unsqueeze(-1)   # (n, 1)
        dim = 1 + torch.arange(embed_dim)               # (d,)

        pi = (dim - dim%2)/embed_dim
        self.encoding[:, 1::2] = torch.sin(pos/(10000**pi[1::2]))
        self.encoding[:, 0::2] = torch.cos(pos/(10000**pi[0::2]))

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state (device, etc...).
        self.register_buffer('positional_encoding', self.encoding, persistent=False)


    def forward(self, input):
        seq_len = input.size(-2)
        return input + self.encoding[:seq_len]






def main():
    pe = PositionalEncoding(seq_len=12, embed_dim=32)    
    x = torch.rand((128,12,32))
    y = pe(x)
    print(y.shape)

    transformer = TransformerEncoder(seq_len=12,
                                     embed_dim=32,
                                     num_layers=4,
                                     num_heads=8,
                                     expansion_factor=4,)
    out = transformer(x)
    print(out.shape)

if __name__=='__main__':
    main()
