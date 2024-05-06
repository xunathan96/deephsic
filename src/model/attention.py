import torch
import torch.nn as nn
from einops import rearrange
__all__=['Attention', 'SelfAttention']


class Attention(nn.Module):
    def __init__(self,
                 dim_k,
                 dim_v,
                 dim_o,
                 in_q = None,
                 in_k = None,
                 in_v = None,
                 num_heads = 1,
                 drop_score = 0.,
                 drop_proj = 0.,
                 ):
        r"""Initializes a Multihead Attention model.

        Args:
            dim_k (int): Dimension of the query/key embedding.
            dim_v (int): Dimension of the value embedding.
            dim_o (int): Dimension of the output projection.
            in_q (int): Dimension of the input query (default: uses dim_k).
            in_k (int): Dimension of the input key (default: uses dim_k).
            in_v (int): Dimension of the input value (default: uses dim_v).
            num_heads (int): Number of parallel attention heads (default: 1). Note that embeddings dim_k, dim_v will be split across num_heads.
            drop_score (float): Dropout probability on each normalized score (default: 0).
            drop_proj (float): Dropout probability on each output embedding (default: 0).
        """
        super().__init__()
        if (dim_k % num_heads != 0) or (dim_v % num_heads != 0):
            raise Exception("Embedding dimensions must be divisible by the number of heads.")
        if in_q is None: in_q = dim_k
        if in_k is None: in_k = dim_k
        if in_v is None: in_v = dim_v

        self.dim_k = self.dim_q = dim_k
        self.dim_v = dim_v
        self.dim_o = dim_o
        self.num_heads = num_heads
        self.head_k = self.head_q = dim_k // num_heads
        self.head_v = dim_v // num_heads

        self.to_Q = nn.Linear(in_q, self.dim_q)
        self.to_K = nn.Linear(in_k, self.dim_k)
        self.to_V = nn.Linear(in_v, self.dim_v)
        self.to_proj = nn.Linear(self.dim_v, self.dim_o)
        self.drop_score = nn.Dropout2d(drop_score)
        self.drop_proj = nn.Dropout1d(drop_proj)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, query, key, value) -> torch.Tensor:
        # to Q,K,V embeddings
        q = self.to_Q(query)    # (b, m, h * head_q)
        k = self.to_K(key)      # (b, n, h * head_k)
        v = self.to_V(value)    # (b, n, h * head_v)

        # split each head
        q = rearrange(q, 'b ... (heads q) -> b heads ... q', heads=self.num_heads)     # (b, h, m, head_q)
        k = rearrange(k, 'b ... (heads k) -> b heads ... k', heads=self.num_heads)     # (b, h, n, head_k)
        v = rearrange(v, 'b ... (heads v) -> b heads ... v', heads=self.num_heads)     # (b, h, n, head_v)

        # attention: Softmax(QK^T/sqrt(dK)) @ V
        score = q @ k.mT / (self.head_k**-0.5)          # (b, h, m, n)
        attn = self.drop_score(self.softmax(score)) @ v # (b, h, m, head_v)

        # concat all heads
        multihead = rearrange(attn, 'b heads ... v -> b ... (heads v)', heads=self.num_heads)   # (b, m, dim_v)
        out = self.drop_proj(self.to_proj(multihead))
        return out



class SelfAttention(Attention):
    
    def __init__(self,
                 input_dim,
                 embed_dim,
                 output_dim,
                 num_heads = 1,
                 dim_k = None,
                 dim_v = None,
                 drop_score = 0,
                 drop_proj = 0,):
        r"""Initializes a Multihead Self-Attention model.

        Args:
            input_dim (int): Dimension of raw input.
            embed_dim (int): Dimension of the (multihead) embedding.
            output_dim (int): Dimension of the output projection.
            num_heads (int): Number of parallel attention heads (default: 1). Note that embed_dim will be split across num_heads.
            dim_k (int): Dimension of the query/key embedding (default: uses embed_dim).
            dim_v (int): Dimension of the value embedding (default: uses embed_dim).
            drop_score (float): Dropout probability on each attention head (default: 0).
            drop_proj (float): Dropout probability on each output embedding (default: 0).
        """
        if dim_k is None: dim_k = embed_dim
        if dim_v is None: dim_v = embed_dim
        super().__init__(dim_k = dim_k,
                         dim_v = dim_v,
                         dim_o = output_dim,
                         in_q = input_dim,
                         in_k = input_dim,
                         in_v = input_dim,
                         num_heads = num_heads,
                         drop_score = drop_score,
                         drop_proj = drop_proj)

    def forward(self, input) -> torch.Tensor:
        return super().forward(input, input, input)



def main():

    # attn = Attention(dim_k=8,
    #                  dim_v=4,
    #                  dim_o=6,
    #                  in_q=12,
    #                  in_k=12,
    #                  in_v=15,
    #                  num_heads=2)

    # query = torch.rand((5,12,2))
    # key = torch.rand((5,12,4))
    # value = torch.rand((5,15,4))
    # out = attn(query.mT, key.mT, value.mT)
    # print(out.shape)   # (5, 2, 6)



    selfattn = SelfAttention(input_dim=12,
                             embed_dim=50,
                             output_dim=10,
                             num_heads=5,
                             drop_score=0.1,
                             drop_proj=0.1)

    input = torch.rand((128, 12, 7))
    out = selfattn(input.mT)
    print(out.shape)    # (128, 7, 10)


if __name__=='__main__':
    main()


