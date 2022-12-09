# ------------------------------------------------------------------------------------
# Modified from VQGAN (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.distributed as dist_fn
from torch.nn import functional as F
from typing import List, Tuple, Optional
from einops import rearrange
import math


class VectorQuantizer(nn.Module):
    """
    Simplified VectorQuantizer in the original VQGAN repository
    """
    def __init__(self, dim: int, n_embed: int, beta: float) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.dim = dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_embed, self.dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embed, 1.0 / self.n_embed)

    def forward(self,
                z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z = rearrange(z, 'b c h w -> b h w c').contiguous()  # [B,C,H,W] -> [B,H,W,C]
        z_flattened = z.view(-1, self.dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self,
                           indices: torch.LongTensor,
                           shape: Optional[List[int]] = None) -> torch.FloatTensor:
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class EMAVectorQuantizer(nn.Module):
    """
    EMAVectorQuantizer
    """
    def __init__(self,
                 dim: int,
                 n_embed: int,
                 beta: float,
                 decay: float = 0.99,
                 eps: float = 1e-5,
                 use_l2_norm: bool = False,
                 restart_unused_codes: bool = False) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.dim = dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.use_l2_norm = use_l2_norm
        self.restart_unused_codes = restart_unused_codes
        self.threshold = 1.0

        embedding = torch.randn(n_embed, dim)
        if (self.use_l2_norm):
            embedding = F.normalize(embedding, p=2.0, dim=1, eps=1e-6)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(self.n_embed))
        self.register_buffer("embedding_avg", embedding.clone())

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / math.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    def forward(self,
                z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z = rearrange(z, 'b c h w -> b h w c').contiguous()  # [B,C,H,W] -> [B,H,W,C]
        z_flattened = z.view(-1, self.dim)
        if (self.use_l2_norm):
            z_flattened = F.normalize(z_flattened, p=2.0, dim=1, eps=1e-6)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, self.embedding).view(z.shape)
        embed_onehot = F.one_hot(min_encoding_indices, self.n_embed).type(z_flattened.dtype)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = embed_onehot.transpose(0, 1) @ z_flattened

            dist_fn.all_reduce(embed_onehot_sum, op=dist_fn.ReduceOp.SUM)
            dist_fn.all_reduce(embed_sum, op=dist_fn.ReduceOp.SUM)

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embedding_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            if self.restart_unused_codes:
                n_vectors = z_flattened.size(0)
                if n_vectors < self.n_embed:
                    vectors = self._tile_with_noise(z_flattened, self.n_embed)
                else:
                    vectors = z_flattened
                n_vectors = vectors.shape[0]
                _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:self.n_embed]

                if dist_fn.is_initialized():
                    dist_fn.broadcast(_vectors_random, 0)

                usage = (self.cluster_size.view(-1, 1) >= 1).float()
                self.embedding_avg.mul_(usage).add_(_vectors_random * (1-usage))
                self.cluster_size.mul_(usage.view(-1))
                self.cluster_size.add_(torch.ones_like(self.cluster_size) * (1-usage).view(-1))

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)

            if (self.use_l2_norm):
                embed_normalized = F.normalize(embed_normalized, p=2.0, dim=1, eps=1e-6)

            self.embedding.data.copy_(embed_normalized)

        diff = self.beta * torch.mean((z_q.detach() - z) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        return z_q, diff, min_encoding_indices

    def get_soft_codes(self,
                       z: torch.FloatTensor,
                       temp=1.0,
                       stochastic=False):

        z = rearrange(z, 'b c h w -> b h w c').contiguous()  # [B,C,H,W] -> [B,H,W,C]
        z_flattened = z.view(-1, self.dim)
        if (self.use_l2_norm):
            z_flattened = F.normalize(z_flattened, p=2.0, dim=1, eps=1e-6)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding, 'n d -> d n'))

        soft_code = F.softmax(-d / temp, dim=1)

        if stochastic:
            soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
            code = torch.multinomial(soft_code_flat, 1)
            code = code.reshape(*soft_code.shape[:-1])
        else:
            # min_encoding_indices
            code = torch.argmin(d, dim=1)

        z_q = F.embedding(code, self.embedding).view(z.shape)

        diff = self.beta * torch.mean((z_q.detach() - z) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        return z_q, diff, code, soft_code

    def get_codebook_entry(self,
                           indices: torch.LongTensor,
                           shape: Optional[List[int]] = None) -> torch.FloatTensor:
        z_q = F.embedding(indices, self.embedding)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q
