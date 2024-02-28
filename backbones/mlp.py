from typing import Optional, List, Tuple

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import OrderedDict

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
        return x.reshape(len(x), -1)
    
class MLP(nn.Module):

    def __init__(self,
                 cont_nums,
                 cat_dims,
                 feat_emb_dim = 128,
                 n_hiddens = 4,
                 emb_dim = 256,
                 dropout = 0.1,
    ) -> None:
        super().__init__()

        self.tokenizer = FeatureTokenizer(emb_dim=feat_emb_dim, cont_nums=cont_nums, cat_dims=cat_dims)
        hidden_layers = nn.ModuleList()
        for _ in range(n_hiddens - 1):
            hidden_layers.append(
                nn.Sequential(
                    OrderedDict([       
                                    ("batch_norm", nn.BatchNorm1d(emb_dim)),
                                    ("activation", nn.LeakyReLU()),
                                    ("dropout", nn.Dropout(dropout)),
                                    ("linear", nn.Linear(emb_dim, emb_dim)),
                                ])
                )
            )

        self.net = nn.Sequential(
            nn.Linear(feat_emb_dim * (cont_nums + len(cat_dims) + 1), emb_dim),
            *hidden_layers,
            # nn.LeakyReLU(),
        )
            
        # self.head = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, out_dim)
        # )

        # for layer in zip(self.net, self.head):
        #     if isinstance(layer, nn.Linear):
        #         nn.init.kaiming_uniform_(layer.weight)
        #         nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x_conts, x_cats = x
        return self.net(self.tokenizer(x_conts, x_cats))