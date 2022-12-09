# ------------------------------------------------------------------------------------
# Modified from minDALL-E and minGPT (https://github.com/karpathy/minGPT)
# Copyright (c) 2020 Andrej Karpathy. All Rights Reserved.
# ------------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn

from typing import Optional, List, Tuple
from torch.nn import functional as F


class GELU(nn.Module):
    def __init__(self, use_approx=False):
        super().__init__()
        self.use_approx = use_approx

    def forward(self, x):
        if self.use_approx:
            return x * torch.sigmoid(1.702 * x)
        else:
            return F.gelu(x)


class MultiHeadSelfAttention(nn.Module):

    def __init__(self,
                 ctx_len: int,
                 embed_dim: int,
                 n_heads: int,
                 resid_pdrop: float,
                 attn_pdrop: float,
                 attn_bias: bool,
                 use_mask: bool = True,
                 parallel_len: int = 0,
                 parallel_type: str = 'parallel',
                 code_level: int = 2):
        super().__init__()
        assert embed_dim % n_heads == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.query = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=attn_bias)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim, attn_bias)

        self.n_heads = n_heads
        self.ctx_len = ctx_len
        self.use_mask = use_mask
        self.parallel_len = parallel_len
        self.parallel_type = parallel_type
        self.code_level = code_level

    def forward(self, x, contexts=None, caching=False, past_kv=None):

        (B, T, C) = x.shape
        if contexts is not None:
            B_ctx, T_ctx, C_ctx = contexts.shape

        if not caching:
            assert past_kv is None

        x = x.transpose(0, 1).contiguous()  # (B, T, C) -> (T, B, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)

        if contexts is not None and past_kv is None:
            contexts = contexts.transpose(0, 1).contiguous()  # (B, T_ctx, C) -> (T_ctx, B, C)
            x_ctx = torch.cat([contexts, x], dim=0)
            T_x_with_ctx = x_ctx.shape[0]

            # k, v size (B*nh, T_ctx, hs)
            k = self.key(x_ctx).view(T_x_with_ctx, B*self.n_heads, C//self.n_heads).transpose(0, 1)
            v = self.value(x_ctx).view(T_x_with_ctx, B*self.n_heads, C//self.n_heads).transpose(0, 1)
        else:
            k = self.key(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)
            v = self.value(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)

        if caching:
            # when context is not None, the context is caching.
            present = torch.stack([k, v])
        else:
            present = None

        if past_kv is not None:
            past_key, past_value = past_kv
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
            T_past = past_key.shape[1]
        else:
            T_past = 0

        # Tensor shape below: query: (B * nh, T, hs) X key: (B * nh, hs, T_past+T) -> (B * nh, T, T_past+T)
        att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))

        if self.use_mask:
            if self.parallel_len == 0:
                if past_kv is None:
                    if contexts is None:
                        # standard attention
                        mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
                        mask = mask.view(1, T, T)
                        att = att.masked_fill(~mask[:, :T, :T], float('-inf'))
                    else:
                        # assume it is cross-attention
                        mask_qk = torch.ones(T, T_ctx, device=q.device, dtype=torch.bool)
                        mask_qq = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
                        mask = torch.cat([mask_qk, mask_qq], dim=-1).unsqueeze(0)
                        att = att.masked_fill(~mask[:, :T, :T_ctx+T], float('-inf'))
                else:
                    # cross attention btw query & key
                    mask_qk = torch.ones(T, T_past, device=q.device, dtype=torch.bool)
                    mask_qq = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
                    mask = torch.cat([mask_qk, mask_qq], dim=-1).unsqueeze(0)
                    att = att.masked_fill(~mask[:, :T, :T_past+T], float('-inf'))

            elif self.parallel_len > 0:
                if self.code_level == 2:
                    if past_kv is None:
                        if contexts is None:
                            # standard attention
                            mask = torch.zeros(T, T, device=q.device, dtype=torch.bool)
                            mask[0, 0] = 1
                            if T > self.parallel_len:
                                winT = self.parallel_len
                                for si in range(0, (T-1)//winT):
                                    mask[1+si*winT:(si+1)*winT+1, 0:winT*(si+1)+1] = 1
                            mask = mask.view(1, T, T)
                            att = att.masked_fill(~mask[:, :T, :T], float('-inf'))
                        else:
                            # assume it is cross-attention

                            # causal for context and query
                            mask_qk = torch.ones(T, T_ctx, device=q.device, dtype=torch.bool)
                            # bidirectional for query, query
                            mask_qq = torch.ones(T, T, device=q.device, dtype=torch.bool)
                            mask = torch.cat([mask_qk, mask_qq], dim=-1).unsqueeze(0)
                            att = att.masked_fill(~mask[:, :T, :T_ctx+T], float('-inf'))
                    else:
                        # cross attention btw query & key
                        mask_qk = torch.ones(T, T_past, device=q.device, dtype=torch.bool)
                        mask_qq = torch.ones(T, T, device=q.device, dtype=torch.bool)
                        mask = torch.cat([mask_qk, mask_qq], dim=-1).unsqueeze(0)
                        att = att.masked_fill(~mask[:, :T, :T_past+T], float('-inf'))

                elif self.code_level == 3:
                    Tm = 1 + 4 + 16
                    mask = torch.zeros(Tm, Tm, device=q.device, dtype=torch.bool)

                    # quad-tree parallel
                    if self.parallel_type == 'tree' or self.parallel_type == 'quad':
                        mask[0, 0] = 1
                        mask[1:1+4, 0:1+4] = 1
                        for i in range(0, 4):
                            mask[1+4+4*i:1+4+4*(i+1), 1+4+4*i:1+4+4*(i+1)] = 1  # level 0 self
                            mask[1+4+4*i:1+4+4*(i+1), 0] = 1  # level 2 - level 0
                            mask[1+4+4*i:1+4+4*(i+1), 1+i] = 1  # level 2 - level 1

                    # full parallel
                    elif self.parallel_type == 'parallel':
                        mask[0, 0] = 1
                        mask[1:1+4, 0:1+4] = 1
                        mask[1+4:1+4+16, 0:1+4+16] = 1

                    mask = mask.view(1, Tm, Tm)

                    if past_kv is None:
                        att = att.masked_fill(~mask[:, :T, :T], float('-inf'))
                    else:
                        att = att.masked_fill(~mask[:, T_past:T_past+T, :T_past+T], float('-inf'))

            else:
                assert False

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.bmm(att, v)  # (B*nh, T, T_past+T) X (B*nh, T_past+T, hs) -> (B*nh, T, hs)
        y = y.transpose(0, 1).contiguous().view(T, B, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        if caching:
            return y.transpose(0, 1).contiguous(), present  # (T, B, C) -> (B, T, C)
        else:
            return y.transpose(0, 1).contiguous()  # (T, B, C) -> (B, T, C)


class MultiHeadCrossAttention(nn.Module):

    def __init__(self,
                 ctx_len: int,
                 embed_dim: int,
                 n_heads: int,
                 resid_pdrop: float,
                 attn_pdrop: float,
                 attn_bias: bool,
                 use_mask: bool = False):
        super().__init__()
        assert embed_dim % n_heads == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.query = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=attn_bias)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim, attn_bias)

        self.n_heads = n_heads
        self.ctx_len = ctx_len
        self.use_mask = use_mask

    def forward(self, x, x_query, caching=False, past_kv=None):

        (B, T, C) = x.shape
        (B, Tq, C) = x_query.shape

        if not caching:
            assert past_kv is None

        x = x.transpose(0, 1).contiguous()  # (B, T, C) -> (T, B, C)
        x_query = x_query.transpose(0, 1).contiguous()  # (B, T, C) -> (T, B, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x_query).view(Tq, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)

        k = self.key(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)
        v = self.value(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)

        if caching:
            # when context is not None, the context is caching.
            present = torch.stack([k, v])
        else:
            present = None

        if past_kv is not None:
            past_key, past_value = past_kv
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
            T_past = past_key.shape[1]
        else:
            T_past = 0

        # Tensor shape below: query: (B * nh, T, hs) X key: (B * nh, hs, T_past+T) -> (B * nh, T, T_past+T)
        att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))

        if self.use_mask:
            if past_kv is None:
                # standard attention
                mask = torch.tril(torch.ones(Tq, T, device=q.device, dtype=torch.bool))
                mask = mask.view(1, Tq, T)
                att = att.masked_fill(~mask[:, :Tq, :T], float('-inf'))
            else:
                # cross attention btw query & key
                mask_qk = torch.ones(Tq, T_past, device=q.device, dtype=torch.bool)
                mask_qq = torch.tril(torch.ones(Tq, T, device=q.device, dtype=torch.bool))
                mask = torch.cat([mask_qk, mask_qq], dim=-1).unsqueeze(0)
                att = att.masked_fill(~mask[:, :Tq, :T_past+T], float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.bmm(att, v)  # (B*nh, T, T_past+T) X (B*nh, T_past+T, hs) -> (B*nh, T, hs)
        y = y.transpose(0, 1).contiguous().view(Tq, B, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        if caching:
            return y.transpose(0, 1).contiguous(), present  # (T, B, C) -> (B, T, C)
        else:
            return y.transpose(0, 1).contiguous()  # (T, B, C) -> (B, T, C)


# masked self-attention with context
class Block(nn.Module):
    def __init__(self,
                 ctx_len: int,
                 embed_dim: int,
                 n_heads: int,
                 mlp_bias: bool,
                 attn_bias: bool,
                 resid_pdrop: bool,
                 attn_pdrop: bool,
                 gelu_use_approx: bool,
                 causal_attn: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadSelfAttention(ctx_len=ctx_len,
                                           embed_dim=embed_dim,
                                           n_heads=n_heads,
                                           attn_pdrop=attn_pdrop,
                                           resid_pdrop=resid_pdrop,
                                           attn_bias=attn_bias,
                                           use_mask=causal_attn)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=mlp_bias),
            GELU(gelu_use_approx),
            nn.Linear(4 * embed_dim, embed_dim, bias=mlp_bias),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, contexts=None):
        x = x + self.attn(self.ln1(x), contexts=contexts)
        x = x + self.mlp(self.ln2(x))
        return x

    def sample(self, x, contexts=None, layer_past=None):
        attn, present = self.attn(self.ln1(x), contexts=contexts, caching=True, past_kv=layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x, present


# self-attention with parallel mask
class ParallelBlock(nn.Module):
    def __init__(self,
                 ctx_len: int,
                 embed_dim: int,
                 n_heads: int,
                 mlp_bias: bool,
                 attn_bias: bool,
                 resid_pdrop: bool,
                 attn_pdrop: bool,
                 gelu_use_approx: bool,
                 parallel_len: int = 0,
                 parallel_type: str = 'parallel',
                 code_level: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadSelfAttention(ctx_len=ctx_len,
                                           embed_dim=embed_dim,
                                           n_heads=n_heads,
                                           attn_pdrop=attn_pdrop,
                                           resid_pdrop=resid_pdrop,
                                           attn_bias=attn_bias,
                                           use_mask=True,
                                           parallel_len=parallel_len,
                                           parallel_type=parallel_type,
                                           code_level=code_level)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=mlp_bias),
            GELU(gelu_use_approx),
            nn.Linear(4 * embed_dim, embed_dim, bias=mlp_bias),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, contexts=None):
        x = x + self.attn(self.ln1(x), contexts=contexts)
        x = x + self.mlp(self.ln2(x))
        return x

    def sample(self, x, contexts=None, layer_past=None):
        attn, present = self.attn(self.ln1(x), contexts=contexts, caching=True, past_kv=layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x, present


# masked self-attention + cross-attention
class DecoderBlock(nn.Module):
    def __init__(self,
                 ctx_len: int,
                 embed_dim: int,
                 n_heads: int,
                 mlp_bias: bool,
                 attn_bias: bool,
                 resid_pdrop: bool,
                 attn_pdrop: bool,
                 gelu_use_approx: bool,
                 causal_attn: bool = True):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)

        self.attn1 = MultiHeadSelfAttention(ctx_len=0,
                                            embed_dim=embed_dim,
                                            n_heads=n_heads,
                                            attn_pdrop=attn_pdrop,
                                            resid_pdrop=resid_pdrop,
                                            attn_bias=attn_bias,
                                            use_mask=causal_attn)

        self.attn2 = MultiHeadCrossAttention(ctx_len=ctx_len,
                                             embed_dim=embed_dim,
                                             n_heads=n_heads,
                                             attn_pdrop=attn_pdrop,
                                             resid_pdrop=resid_pdrop,
                                             attn_bias=attn_bias,
                                             use_mask=False)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=mlp_bias),
            GELU(gelu_use_approx),
            nn.Linear(4 * embed_dim, embed_dim, bias=mlp_bias),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, contexts=None):
        x = x + self.attn1(self.ln1(x))
        x = x + self.attn2(contexts, self.ln2(x))
        x = x + self.mlp(self.ln3(x))
        return x

    def sample(self, x, contexts=None, layer_past=None):
        attn1, present = self.attn1(self.ln1(x), caching=True, past_kv=layer_past)
        x = x + attn1

        x = x + self.attn2(contexts, self.ln2(x))
        x = x + self.mlp(self.ln3(x))
        return x, present