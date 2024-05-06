__author__ = 'Nathaniel Xu'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple
from pathlib import Path
from collections.abc import Iterable
from typing import Union

class SegFormer(nn.Module):
    """ MiT Backbone + All-MLP Segmentation Head """
    def __init__(self,
                 img_size=(512,512),
                 in_channels=3,
                 embed_dims=(32,64,160,256),        # MiT encoder embedding dimension per stage
                 patch_sizes=(7,3,3,3),             # overlapping patch kernel sizes per stage
                 patch_strides=(4,2,2,2),           # overlapping patch strides
                 num_layers=(2,2,2,2),              # number of Attention+MixFFN layers per stage
                 num_heads=(1,2,5,8),               # number of attention heads per stage
                 reduction_ratios=(8,4,2,1),        # reduction ratio for efficient self-attention per stage
                 expansion_factors=(8,8,4,4),       # expansion factor for the latent feature size the in MixFFN block per stage
                 decoder_dim=256,                   # all-MLP decoder latent dimension size
                 num_classes=10,                    # number of predicted classes
                 drop_path=0):                      # the stochastic depth (drop path) rate
        super().__init__()

        # assert paramters are specified for each stage, and embedding dimensions are divisible by the number of self-attention heads 
        assert all(len(embed_dims)==len(t) for t in (patch_sizes, patch_strides, num_layers, num_heads, reduction_ratios, expansion_factors))
        assert all(dim%heads==0 for (dim, heads) in zip(embed_dims, num_heads))

        # MiT backbone
        self.mit = MiT(in_channels,
                       embed_dims,
                       patch_sizes,
                       patch_strides,
                       num_layers,
                       num_heads,
                       reduction_ratios,
                       expansion_factors,
                       drop_path)

        # segmentation head
        self.head = MLPSegmentationHead(embed_dims,
                                        decoder_dim,
                                        upsample_size=(img_size[0]//4, img_size[1]//4),
                                        num_classes=num_classes)    

    def forward(self, x):
        B,C,H,W = x.shape
        fmaps = self.mit(x)     # [(B,Di,H/s,W/s)] * stages
        
        #for i,f in enumerate(fmaps):
        #    print(f'f{i+1}', f.shape)

        mask = self.head(fmaps)
        mask = F.interpolate(mask, size=(H,W), mode='bilinear', align_corners=False)    # upscale to original size
        return mask


    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path, device: torch.device = torch.device('cpu')):
        self.load_state_dict(torch.load(path, map_location=device))


"""
model = SegFormer(in_channels = 3,
                    embed_dims = (32,64,160,256),
                    num_layers = (2,2,2,2),
                    num_heads = (1,2,5,8),
                    reduction_ratios = (8,4,2,1),
                    expansion_factors = (8,8,4,4),
                    decoder_dim = 256,
                    num_classes = 10,
                    drop_path = 0.1)
x = torch.randn(2, 3, 512, 512).to(self.device)
mask = self.model(x)    # (B, C=15, 512, 512)
print('out.shape:', mask.shape)
"""




class MLPSegmentationHead(nn.Module):
    def __init__(self,
                 embed_dims,
                 decoder_dim,
                 upsample_size,
                 num_classes,
                 drop_rate=0.):
        super().__init__()
        
        self.rescale = nn.ModuleDict()
        for dim in embed_dims:
            self.rescale[str(dim)] = nn.Sequential(nn.Conv2d(dim, decoder_dim, kernel_size=1),
                                                   nn.Upsample(upsample_size, mode='bilinear', align_corners=False))

        self.to_segmentation = nn.Sequential(nn.Conv2d(len(embed_dims)*decoder_dim, decoder_dim, kernel_size=1),
                                             nn.BatchNorm2d(decoder_dim),
                                             nn.Dropout2d(drop_rate),
                                             nn.Conv2d(decoder_dim, num_classes, kernel_size=1))

    def forward(self, features):
        scaled_features = []
        for x in features:
            B,C,H,W = x.shape
            x = self.rescale[str(C)](x)         # (B, decoder_dim, img_H/4, img_W/4)
            scaled_features.append(x)

        x = torch.cat(scaled_features, dim=1)   # (B, n_stages*decoder_dim, img_H/4, img_W/4)
        x = self.to_segmentation(x)             # (B, num_classes, img_H/4, img_W/4)
        return x



class MiT(nn.Module):
    """ (Overlap Patch Embed + Transformer Block) * M """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 patch_sizes,
                 patch_strides,
                 num_layers,
                 num_heads,
                 reduction_ratios,
                 expansion_factors,
                 drop_path=0.):
        super().__init__()

        # assert paramters are specified for each stage, and embedding dimensions are divisible by the number of self-attention heads 
        assert all(len(embed_dims)==len(t) for t in (patch_sizes, patch_strides, num_layers, num_heads, reduction_ratios, expansion_factors))
        assert all(dim%heads==0 for (dim, heads) in zip(embed_dims, num_heads))
        
        num_stages = len(embed_dims)
        embed_dims = (in_channels, *embed_dims)
        dpr = np.linspace(0, drop_path, sum(num_layers))

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            dpr_per_layer = dpr[sum(num_layers[:i]):sum(num_layers[:i+1])]
            block = nn.Sequential(OverlapPatchEmbedding(patch_size=patch_sizes[i],
                                                        stride=patch_strides[i],
                                                        in_channels=embed_dims[i],
                                                        embed_dim=embed_dims[i+1]),
                                  TransformerBlock(num_layers=num_layers[i],
                                                   in_features=embed_dims[i+1],
                                                   num_heads=num_heads[i],
                                                   reduction_ratio=reduction_ratios[i],
                                                   expansion_factor=expansion_factors[i],
                                                   drop_path_rates=dpr_per_layer))
            self.stages.append(block)

    def forward(self, x):
        B,C,H,W = x.shape
        latents = []
        for stage in self.stages:
            x = stage(x)
            latents.append(x)
        return latents  # [(B,D1,H/4,W/4), (B,D2,H/8,W/8), (B,D3,H/16,W/16), (B,D4,H/32,W/32)]



class TransformerBlock(nn.Module):
    """ (Self-Attention + Mix-FFN) * N """
    def __init__(self,
                 num_layers,
                 in_features,
                 num_heads,
                 reduction_ratio=1,
                 expansion_factor=4,
                 drop_attn=0.,
                 drop_proj=0.,
                 drop_ffn=0.,
                 drop_path_rates=0.):
        super().__init__()

        if not isinstance(drop_path_rates, Iterable):
            drop_path_rates = (drop_path_rates,)*num_layers
        assert len(drop_path_rates)==num_layers, f"the number of drop path rates {drop_path_rates} should be equal to the number of layers {num_layers}"
        assert in_features%num_heads==0, f"feature dimension {in_features} should be divisible by num_heads {num_heads}."
        dim_k = dim_v = in_features//num_heads

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            attn = nn.Sequential(ChannelLayerNorm(in_features),
                                 MultiHeadAttention(in_features,
                                                    dim_k,
                                                    dim_v,
                                                    num_heads,
                                                    drop_attn,
                                                    drop_proj,
                                                    reduction_ratio),
                                 DropPath(drop_path_rates[i]) if drop_path_rates[i] > 0. else nn.Identity())
            ffn = nn.Sequential(ChannelLayerNorm(in_features),
                                MixFeedForward(in_features,
                                               hidden_features=expansion_factor*in_features,
                                               drop=drop_ffn),
                                DropPath(drop_path_rates[i]) if drop_path_rates[i] > 0. else nn.Identity())
            self.layers.append(nn.ModuleList([attn, ffn]))
    
    def forward(self, x):
        B,D,H,W = x.shape
        for (attn, ffn) in self.layers:
            x = attn(x) + x
            x = ffn(x) + x
        return x    # (B, D, H, W)



class MultiHeadAttention(nn.Module):
    """ Multi-Head Self-Attention with Sequence Reduction """
    def __init__(self,
                 dim_in,
                 dim_k,
                 dim_v,
                 num_heads,
                 drop_attn=0.,
                 drop_proj=0.,
                 reduction_ratio=1):
        super().__init__()

        self.dim_in = dim_in
        self.dim_k = self.dim_q = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio

        self.to_q = nn.Conv2d(dim_in, num_heads*dim_k, kernel_size=1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, num_heads*(dim_k+dim_v), kernel_size=reduction_ratio, stride=reduction_ratio, bias=False)
        self.to_proj = nn.Conv2d(num_heads*dim_v, dim_in, kernel_size=1, bias=False)
        self.drop_attn = nn.Dropout2d(drop_attn)
        self.drop_proj = nn.Dropout2d(drop_proj)
        # NOTE: The original implementation uses layernorm after sequence reduction

    def forward(self, x):
        B,D,H,W = x.shape
        # embed input into Q, K, V
        q = self.to_q(x)        # (B, heads*Dk, H, W)
        kv = self.to_kv(x)      # (B, heads*(Dk+Dv), H/r, W/r)
        k,v = kv.split([self.num_heads*self.dim_k, self.num_heads*self.dim_v], dim=1)

        # split into heads
        q, k, v = map(lambda t: rearrange(t, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads), (q, k, v))    # TODO: verify its same as view + transpose

        # attention: Softmax(QK^T/sqrt(dK)) @ V
        dot_prod = q @ k.mT / (self.dim_k**-0.5)            # (B, heads, H*W, H*W/r2)
        rank = self.drop_attn(dot_prod.softmax(dim=-1))     # (B, heads, H*W, H*W/r2)
        attn = rank @ v                                     # (B, heads, H*W, Dv)

        # output head
        out = rearrange(attn, 'b heads (h w) d -> b (heads d) h w', h=H)    # (B, heads*Dv, H, W)
        out = self.drop_proj(self.to_proj(out))                             # (B, D, H, W)
        return out



class MixFeedForward(nn.Module):
    """ MLP(GELU(DWConv3x3(MLP(x)))) """
    # we use Conv1x1 instead of linear since each channel is the latent dimension and H*W is the batch number
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.mff = nn.Sequential(nn.Conv2d(in_features, hidden_features, kernel_size=1),
                                 DWConv2d(hidden_features, hidden_features, kernel_size=3, padding=1),
                                 nn.GELU(),
                                 nn.Dropout2d(drop),
                                 nn.Conv2d(hidden_features, out_features, kernel_size=1),
                                 nn.Dropout2d(drop))
    def forward(self, x):
        return self.mff(x)  # (B, D, H, W)



class DWConv2d(nn.Module):
    """ Depth-Wise Separable Convolution """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.dwconv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
                                    nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        return self.dwconv(x)   # (B, Dout, H, W)



class OverlapPatchEmbedding(nn.Module):
    """ Strided Conv + LayerNorm """
    def __init__(self,
                 patch_size,
                 stride,
                 in_channels,
                 embed_dim):
        super().__init__()
        patch_size = to_2tuple(patch_size) if not isinstance(patch_size, tuple) else patch_size
        self.embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=stride, padding=(patch_size[0]//2, patch_size[1]//2))
        #self.embed = nn.Sequential(nn.Conv2d(in_channels, embed_dim, patch_size, stride=stride, padding=(patch_size[0]//2, patch_size[1]//2)),
        #                           ChannelLayerNorm(embed_dim))
        # NOTE: original implementation contains layernorm here. However, the subsequent Transformer block also starts with a layernorm.

    def forward(self, x):
        return self.embed(x)    # (B, embed_dim, H//[4,8,16,32], W//[4,8,16,32])





# ----------------------
#        HELPERS
# ----------------------

class ChannelLayerNorm(nn.LayerNorm):
    """ Layer norm applied along the channel dimension of an (B, C, H, W) image tensor """
    def forward(self, x):
        B,D,H,W = x.shape
        x = x.flatten(-2).transpose(1,2)
        x = nn.LayerNorm.forward(self, x)
        x = x.transpose(1,2).view(B,D,H,W)
        return x

class ReshapeImg2Embed(nn.Module):
    """ Reshapes a (B, C, H, W) image tensor into a (B, H*W, C) embedding """
    def forward(self, x):
        return x.flatten(-2).permute(0,2,1)

class ReshapeEmbed2Img(nn.Module):
    """ Reshapes a (B, H*W, C) embedding into a (B, C, H, W) image tensor """
    def forward(self, x, H, W):
        B, HW, C = x.shape
        return x.transpose(1,2).view(B, C, H, W)

