from typing import Optional, List, Tuple, Dict

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class FeatureTokenizer(nn.Module):

    def __init__(self, 
                emb_dim : int = None,
                cont_nums : int = None,
                cat_dims : Tuple[int] = None
        ) -> None:
        super().__init__()
        emb_dim = emb_dim
        cont_nums = cont_nums
        cat_dims = cat_dims

        bias_dim = 0

        if cont_nums is not None:
            bias_dim += cont_nums

        if cat_dims is not None:
            bias_dim += len(cat_dims)

            category_offsets = torch.tensor([0] + cat_dims[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)

            self.cat_weights = nn.Embedding(sum(cat_dims), emb_dim)
        
        self.weight = nn.Parameter(torch.Tensor(cont_nums + 1, emb_dim))
        self.bias = nn.Parameter(torch.Tensor(bias_dim, emb_dim))

    def forward(self, 
                x_conts: torch.Tensor = None, 
                x_cats: torch.Tensor = None
        ) -> torch.Tensor:
        
        x_conts = torch.cat(
            [torch.ones(
                len(x_conts) if x_conts is not None else len(x_cats), 1, 
                device = x_conts.device if x_conts is not None else x_cats.device)
            ]
            + ([] if x_conts is None else [x_conts]),
            dim = 1
        )

        x = self.weight[None] * x_conts[:, :, None]

        if x_cats is not None:

            x = torch.cat(
                [x, self.cat_weights(x_cats + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x

class MultiheadAttention(nn.Module):
    def __init__(self,
                 emb_dim : int,
                 n_heads : int,
                 attn_dropout : float,
    ) -> None:
        super().__init__()
        
        assert emb_dim % n_heads == 0

        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads

        self.W_qkv = nn.Linear(emb_dim, emb_dim * 3)

        self.W_out = nn.Linear(emb_dim, emb_dim)
        
        self.scale = emb_dim ** - 0.5

        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

    
    def forward(self,
                x: torch.Tensor,
                return_attn: bool = False
    ) -> torch.Tensor:
        
        batch_size = len(x)

        
        qkv = self.W_qkv(x)
        qkv = qkv.reshape(batch_size, -1, self.n_heads, self.head_dim * 3)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim = -1)
        
        # print(q.shape, k.shape)
        attention = F.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1)
        attention = F.softmax(attention, dim = -1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        v = attention @ v

        v = v.permute(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.emb_dim)

        out = self.W_out(v)

        if return_attn:
            return out, attention
        else:
            return out
        
def reglu(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

class Transformer(nn.Module):

    def __init__(self,
                 emb_dim: int, 
                 n_heads: int, 
                 attn_dropout: float,
                 ffn_dropout: float,
                 ffn_factor_dim: float,
                 layer_idx: int,
    ) -> None:
        super().__init__()

        hidden_dim = int(emb_dim * ffn_factor_dim)

        self.block = nn.ModuleDict(
            {
                'attention' : MultiheadAttention(emb_dim, n_heads, attn_dropout),
                'linear0' : nn.Linear(emb_dim, hidden_dim * 2),
                'linear1' : nn.Linear(hidden_dim, emb_dim),
                'norm1': nn.LayerNorm(emb_dim)
            }
        )
        if layer_idx:
            self.block['norm0'] = nn.LayerNorm(emb_dim)
        if ffn_dropout > 0:
            self.block['ffn_dropout'] = nn.Dropout(ffn_dropout)

        self.activation = reglu

    def forward(self,
                x: torch.Tensor
    ) -> torch.Tensor:

        if 'norm0' in self.block:
            x_res = self.block['attention'](self.block['norm0'](x))
        else:
            x_res = self.block['attention'](x)
        
        x = x + x_res

        x_res = self.block['norm1'](x)
        x_res = self.block['linear0'](x_res)
        x_res = self.activation(x_res)
        x_res = self.block['ffn_dropout'](x_res)
        x_res = self.block['linear1'](x_res)

        x = x + x_res

        return x


class FTTransformer(nn.Module):

    def __init__(self,
                 cont_nums: int,
                 cat_dims: int,
                 emb_dim: int, 
                 n_heads: int, 
                 attn_dropout: float,
                 ffn_dropout: float,
                 ffn_factor_dim: float,
                 depth: int,
                 out_dim: int,
    ) -> None:
        super().__init__()

        self.tokenizer = FeatureTokenizer(emb_dim, cont_nums, cat_dims)

        self.backbone = nn.Sequential(
            *[
                Transformer(
                    emb_dim,
                    n_heads, 
                    attn_dropout, 
                    ffn_dropout, 
                    ffn_factor_dim,
                    idx
                )
                for idx in range(depth)
            ],
        )
        
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.xavier_uniform_(module.weight)
    #         module.bias.data.fill_(0.01)
            
            
    def forward(self,
                x: Tuple[torch.Tensor]
    ) -> torch.Tensor:

        x_conts, x_cats = x

        x = self.tokenizer(x_conts, x_cats)

        x = self.backbone(x)

        x = x[:, 0]

        return x

        

        




            






        
        


        