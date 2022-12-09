# ------------------------------------------------------------------------------------
# HQ-Transformer for Multiscale Modeling
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import math
import copy
import torch
import torch.nn as nn

from typing import Optional, Tuple, List
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.nn import functional as F
from einops import rearrange

from hqvae.models.stage2.layers import Block, ParallelBlock
from hqvae.utils.sampling import cutoff_topk_logits, cutoff_topp_probs, get_positional_encoding


class HQTransformer(nn.Module):

    def initialize_embed(self, hparams):
        # input embedding
        self.spatial_embedding = hparams.embedding_type

        if hparams.embedding_type == 'reduce':
            embed_dim = hparams.embed_dim
            self.tok_emb_levels = nn.ModuleList()
            for vocab_size in self.vocab_sizes:
                self.tok_emb_levels.append(nn.Embedding(vocab_size, embed_dim))
                embed_dim /= 4

        elif 'transformer' in hparams.embedding_type:
            tok = 'transformer'
            self.spatial_embedding = tok
            n_layers_emb = int(hparams.embedding_type.split(tok)[-1])
            self.tok_emb_levels = nn.ModuleList()
            for vocab_size in self.vocab_sizes:
                self.tok_emb_levels.append(nn.Embedding(vocab_size, hparams.embed_dim))
            self.pos_emb_emb = nn.Embedding(self.code_len, self.pos_emb_dim)

            self.emb_blocks = [Block(ctx_len=self.code_len,
                                     embed_dim=hparams.embed_dim,
                                     n_heads=hparams.n_heads,
                                     mlp_bias=hparams.mlp_bias,
                                     attn_bias=hparams.attn_bias,
                                     resid_pdrop=hparams.resid_pdrop,
                                     attn_pdrop=hparams.attn_pdrop,
                                     gelu_use_approx=hparams.gelu_use_approx,
                                     causal_attn=False) for i in range(1, n_layers_emb)]
            self.emb_blocks = nn.Sequential(*self.emb_blocks)

        else:
            assert False

    def initialize_body(self, hparams):
        # sos token embedding
        if self.use_cls_cond:
            self.sos = nn.Embedding(hparams.n_classes, hparams.embed_dim)
            self.idx_pred = 0

        elif self.use_txt_cond:
            self.tok_emb_txt = nn.Embedding(self.vocab_size_txt, hparams.embed_dim)
            self.pos_emb_txt = nn.Embedding(hparams.ctx_len_txt, hparams.embed_dim)

            self.head_txt = nn.Linear(hparams.embed_dim, self.vocab_size_txt, bias=False)
            self.ln_txt = nn.LayerNorm(hparams.embed_dim)

            self.idx_pred = hparams.ctx_len_txt

        else:
            self.sos = nn.Parameter(torch.randn(1, 1, hparams.embed_dim))
            self.idx_pred = 0

        # position embedding with the first level code (top code)
        self.position_embedding = hparams.position_embedding
        if self.position_embedding == '1d':
            self.pos_emb_top = nn.Embedding(hparams.ctx_len_img, self.pos_emb_dim)
        elif self.position_embedding == '2d':
            H = int(math.sqrt(hparams.ctx_len_img))
            self.pos_emb_top_h = nn.Embedding(H, self.pos_emb_dim)
            self.pos_emb_top_w = nn.Embedding(H, self.pos_emb_dim)

        self.drop = nn.Dropout(hparams.embd_pdrop)

        # transformer blocks
        self.blocks = [Block(ctx_len=hparams.ctx_len_img + 1,
                             embed_dim=hparams.embed_dim,
                             n_heads=hparams.n_heads,
                             mlp_bias=hparams.mlp_bias,
                             attn_bias=hparams.attn_bias,
                             resid_pdrop=hparams.resid_pdrop,
                             attn_pdrop=hparams.attn_pdrop,
                             gelu_use_approx=hparams.gelu_use_approx) for i in range(1, hparams.n_layers+1)]
        self.blocks = nn.Sequential(*self.blocks)

        self.ln_f = nn.LayerNorm(hparams.embed_dim)

    def initialize_head(self, hparams, hparams_dec):
        # sos token embedding
        self.sos_depth = nn.Parameter(torch.randn(1, 1, hparams_dec.embed_dim))

        # input embedding
        self.depth_embedding = 'baseline'
        self.tok_emb_depth_levels = nn.ModuleList()
        for li, vocab_size in enumerate(self.vocab_sizes):
            if('reduce' in self.decoding_type):
                if(li == 2):
                    chn_mult = 16
                else:
                    chn_mult = 4
                self.tok_emb_depth_levels.append(nn.Embedding(vocab_size, chn_mult*hparams.embed_dim))

            else:
                self.tok_emb_depth_levels.append(nn.Embedding(vocab_size, hparams.embed_dim))

        self.pos_emb_depths = nn.ModuleList()

        # legacy
        if 'tree' in self.decoding_type or self.decoding_type == 'old-parallel':
            for vocab_size in self.vocab_sizes:
                self.pos_emb_depths.append(nn.Embedding(4, hparams_dec.embed_dim))
        elif 'parallel' in self.decoding_type:
            if len(self.vocab_sizes) == 3:
                self.pos_emb_depths.append(nn.Embedding(4, hparams_dec.embed_dim))
                self.pos_emb_depths.append(nn.Embedding(16, hparams_dec.embed_dim))
            else:
                assert False  # not supported currently.
        elif 'top2mid2bot' in self.decoding_type:
            if len(self.vocab_sizes) == 3:
                self.pos_emb_depths.append(nn.Embedding(1+4+16, hparams_dec.embed_dim))
            else:
                assert False  # not supported currently.

        # depth transformer blocks
        if 'top2mid2bot' in self.decoding_type:
            self.depths = [Block(ctx_len=1+4+16,
                                 embed_dim=hparams_dec.embed_dim,
                                 n_heads=hparams_dec.n_heads,
                                 mlp_bias=hparams_dec.mlp_bias,
                                 attn_bias=hparams_dec.attn_bias,
                                 resid_pdrop=hparams_dec.resid_pdrop,
                                 attn_pdrop=hparams_dec.attn_pdrop,
                                 gelu_use_approx=hparams_dec.gelu_use_approx) for i in range(1, hparams_dec.n_layers+1)]
        else:
            self.depths = [ParallelBlock(ctx_len=self.code_len,
                                         embed_dim=hparams_dec.embed_dim,
                                         n_heads=hparams_dec.n_heads,
                                         mlp_bias=hparams_dec.mlp_bias,
                                         attn_bias=hparams_dec.attn_bias,
                                         resid_pdrop=hparams_dec.resid_pdrop,
                                         attn_pdrop=hparams_dec.attn_pdrop,
                                         gelu_use_approx=hparams_dec.gelu_use_approx,
                                         parallel_len=4,
                                         parallel_type=self.decoding_type.split('-')[0],
                                         code_level=self.code_level) for i in range(1, hparams_dec.n_layers+1)]
        self.depths = nn.Sequential(*self.depths)

        # head
        self.ln_levels = nn.ModuleList()
        self.head_levels = nn.ModuleList()
        for vocab_size in self.vocab_sizes:
            self.ln_levels.append(nn.LayerNorm(hparams_dec.embed_dim))
            self.head_levels.append(nn.Linear(hparams_dec.embed_dim, vocab_size, bias=False))

    def __init__(self,
                 vocab_sizes: List[int],
                 vocab_size_txt: int,
                 decoding_type: str,
                 use_cls_cond: bool,
                 use_txt_cond: bool,
                 hparams: OmegaConf,
                 hparams_dec: OmegaConf = None) -> None:

        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.vocab_size_txt = vocab_size_txt
        self.use_cls_cond = use_cls_cond
        self.use_txt_cond = use_txt_cond
        self.ctx_len_img = hparams.ctx_len_img

        self.pos_emb_dim = hparams.embed_dim

        self.code_level = len(vocab_sizes)

        code_len = 1
        if self.code_level > 1:
            code_len += 2 ** 2
        if self.code_level > 2:
            code_len += 4 ** 2
        if self.code_level > 3:
            code_len += 8 ** 2
        self.code_len = code_len
        self.num_pairs = 4

        if decoding_type is None:
            self.decoding_type = 'tree'
        else:
            self.decoding_type = decoding_type

        self.n_layers = hparams.n_layers
        if hparams_dec is None:
            print('hparam_dec is None. Use hparam instead.')
            hparams_dec = copy.deepcopy(hparams)
            hparams_dec.n_layers = 4  # same with depth trans. in RQ

        self.n_layers_depth = hparams_dec.n_layers

        self.initialize_embed(hparams)
        self.initialize_body(hparams)
        self.initialize_head(hparams, hparams_dec)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,
                codes: Tuple[torch.LongTensor],
                labels: Optional[torch.LongTensor] = None,
                model_stage1: Optional[torch.nn.Module] = None) -> torch.FloatTensor:

        # main transformer
        h = self.forward_embeddings(codes, labels, model_stage1)

        # depth transformer
        if 'top2mid2bot' in self.decoding_type:
            return self.forward_causal(h, codes, model_stage1)
        else:
            return self.forward_hierarchy(h, codes, model_stage1)

    def forward_embeddings(self,
                           codes: Tuple[torch.LongTensor],
                           labels: Optional[torch.LongTensor] = None,
                           model_stage1: Optional[torch.nn.Module] = None):
        top_codes, bot_codes = codes[0], codes[-1]

        B, Ttop = top_codes.shape
        B, Tbot = bot_codes.shape
        Htop = int(math.sqrt(Ttop))

        # main transformer
        # single token = single top code [1] + top_pos +  four bot codes [1 3]
        #                                                                [2 4]
        if self.position_embedding == '1d':
            xps = torch.arange(Ttop, device=top_codes.device).repeat((B, 1))
            pos_emb = self.pos_emb_top(xps)

        elif self.position_embedding == '2d':
            xs_pos_h = torch.arange(Htop, device=top_codes.device).repeat(B, Htop, 1).transpose(1, 2)
            xs_pos_w = torch.arange(Htop, device=top_codes.device).repeat(B, Htop, 1)
            pos_emb_h = self.pos_emb_top_h(xs_pos_h)
            pos_emb_w = self.pos_emb_top_w(xs_pos_w)
            pos_emb = pos_emb_h + pos_emb_w
            pos_emb = rearrange(pos_emb, 'B H W C -> B (H W) C')

        if self.spatial_embedding == 'transformer':
            emb_level0 = self.tok_emb_levels[0](codes[0])
            emb_level0 += pos_emb
            emb_level0 = rearrange(emb_level0, 'B L (U K) -> (B L) U K', U=1)
            hs = [emb_level0]

            emb_level1 = self.tok_emb_levels[1](codes[1])
            emb_level1 = rearrange(emb_level1, 'B (H H2 W W2) K -> (B H W) (H2 W2) K', H2=2, W2=2, H=Htop, W=Htop)
            hs.append(emb_level1)

            if self.code_level > 2:
                emb_level2 = self.tok_emb_levels[2](codes[2])
                emb_level2 = rearrange(emb_level2, 'B (H H2 W W2) K -> (B H W) (H2 W2) K', H2=4, W2=4, H=Htop, W=Htop)
                hs.append(emb_level2)

            h = torch.cat(hs, dim=1)
            xps_embed = torch.arange(self.code_len, device=top_codes.device)
            h += self.pos_emb_emb(xps_embed).unsqueeze(0)
            h = self.emb_blocks(h)
            h = h.mean(dim=1)
            h = rearrange(h, '(B L) K -> B L K', B=B)
        else:
            assert False

        if self.use_cls_cond:
            sos = self.sos(labels).unsqueeze(1)
        elif self.use_txt_cond:
            pos_txt = torch.arange(0, self.idx_pred, device=top_codes.device).unsqueeze(0)
            sos = self.tok_emb_txt(labels)
            sos += self.pos_emb_txt(pos_txt)
        else:
            sos = self.sos.repeat((B, 1, 1))

        h = torch.cat([sos, h[:, :-1]], dim=1).contiguous()

        h = self.drop(h)
        h = self.blocks(h)
        h = self.ln_f(h)

        return h

    def forward_hierarchy(self,
                          h: torch.FloatTensor,
                          codes: Tuple[torch.LongTensor],
                          model_stage1: Optional[torch.nn.Module] = None):

        top_codes, bot_codes = codes[0], codes[-1]

        B, Ttop = top_codes.shape
        B, Tbot = bot_codes.shape
        Htop = int(math.sqrt(Ttop))

        # (B, H, W, HW) batch size, top code height, width, top code height*width
        # (2H, 2W) middle code height, width
        # (4H, 4W) bottom code height, width

        if self.use_txt_cond:
            h_txt = h[:, :self.idx_pred-1, :]
            logits_txt = self.head_txt(self.ln_txt(h_txt))

            h = h[:, self.idx_pred-1:, :]

        # level 0
        # output code     [Top0]
        # input embedding [h+sos]
        # BHW x 1 x K
        sos_depth = self.sos_depth.repeat((B*Ttop, 1, 1))
        h = rearrange(h, 'B HW (HW1 K) -> (B HW) HW1 K', HW1=1) + sos_depth
        hs = [h]

        # level 1
        # output code     [Mid0,      Mid1,      Mid2,      Mid3,    ]
        # input embedding [Top0+Pos0, Top0+Pos1, Top0+Pos2, Top0+Pos3]
        xps_level0_depth = torch.arange(4, device=top_codes.device).repeat((B*Ttop, 1))
        pos_level0_depth = self.pos_emb_depths[0](xps_level0_depth)

        if 'reduce' in self.decoding_type:
            top_embed = self.tok_emb_depth_levels[0](codes[0])
            top_embed = rearrange(top_embed, 'B HW (HW1 K) -> (B HW) HW1 K', HW1=4)
        else:
            top_embed = self.tok_emb_depth_levels[0](codes[0])
            top_embed = rearrange(top_embed, 'B HW (HW1 K) -> (B HW) HW1 K', HW1=1)

        # broadcasting top codes (except the case for reduce)
        # BHW x 4 x K = BHW x 1 x K + BHW x 4 x K
        emb_level0 = top_embed + pos_level0_depth
        hs.append(emb_level0)

        # level 2
        # output code     [Bot0,      Bot1,      ..., Bot15]
        # input embedding [Mid0+Pos0, Mid0+Pos1, ..., Mid3+Pos15] (+ Top0 if head embedding add case)
        if self.code_level > 2:
            emb_level1 = self.tok_emb_depth_levels[1](codes[1])
            if 'parallel' in self.decoding_type:

                # position embedding in head for bottom code
                xps_level1_depth = torch.arange(16, device=top_codes.device).repeat((B*Ttop, 1))
                pos_level1_depth = self.pos_emb_depths[1](xps_level1_depth)
                reorder_into_pyramid = 'BHW (H1 H2 W1 W2) K -> BHW (H1 W1) (H2 W2) K'
                pos_level1_depth = rearrange(pos_level1_depth, reorder_into_pyramid, H1=2, W1=2, H2=2, W2=2)

                if 'reduce' in self.decoding_type:
                    global_to_pyramid_split_chns = 'B (H H1 W W1) (K1 K) -> (B H W) (H1 W1) K1 K'
                    emb_level1 = rearrange(emb_level1, global_to_pyramid_split_chns, H1=2, W1=2, K1=4, H=Htop, W=Htop)
                else:
                    global_to_pyramid = 'B (H H1 H2 W W1 W2) K -> (B H W) (H1 W1) (H2 W2) K'
                    emb_level1 = rearrange(emb_level1, global_to_pyramid, H1=2, W1=2, H2=1, W2=1, H=Htop, W=Htop)

                # broadcasting middle codes (except the case for reduce)
                # BHW x 4 x 4 x K = BHW x 4 x 1 x K + BHW x 4 x 4 x K
                emb_level1 = emb_level1 + pos_level1_depth

                flatten_pyramid = 'BHW (H1 W1) (H2 W2) K -> BHW (H1 H2 W1 W2) K'
                emb_level1 = rearrange(emb_level1, flatten_pyramid, H1=2, W1=2, H2=2, W2=2)

            else:
                assert False

            if 'add' in self.decoding_type:
                emb_level1 = emb_level1 + top_embed

                hs.append(emb_level1)

        h = torch.cat(hs, dim=1)
        h = self.depths(h)

        logits_level0 = self.head_levels[0](self.ln_levels[0](h[:, 0, :]))
        logits_level1 = self.head_levels[1](self.ln_levels[1](h[:, 1:(1+4), :]))

        logits = []
        logits.append(rearrange(logits_level0, '(B HW) K -> B HW K', B=B))
        logits.append(rearrange(logits_level1, '(B H W) (H1 W1) K -> B (H H1 W W1) K', H1=2, W1=2, H=Htop, W=Htop))

        if self.code_level > 2:
            logits_level2 = self.head_levels[2](self.ln_levels[2](h[:, (1+4):(1+4+16), :]))
            if 'parallel' in self.decoding_type:
                pyramid_to_global = '(B H W) (H1 H2 W1 W2) K -> B (H H1 H2 W W1 W2) K'
                logits.append(rearrange(logits_level2, pyramid_to_global, H1=2, W1=2, H2=2, W2=2, H=Htop, W=Htop))

        if self.use_txt_cond:
            logits.append(logits_txt)

        return logits

    @torch.no_grad()
    def sampling_step(self,
                      sos: torch.FloatTensor,
                      codes: List[torch.LongTensor],
                      pos_codes: torch.LongTensor,
                      use_fp16: bool = True,
                      top_k: Optional[List[float]] = None,
                      top_p: Optional[List[float]] = None,
                      softmax_temperature: List[float] = [1.0, 1.0, 1.0],
                      past: Optional[torch.Tensor] = None,
                      model_stage1: Optional[torch.nn.Module] = None
                      ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:

        with autocast(enabled=use_fp16):
            hs, presents = self.sampling_step_spatial(sos, codes, pos_codes, past, model_stage1)

            if 'top2mid2bot' in self.decoding_type:
                codes = self.sampling_depth_causal(hs, top_k, top_p, softmax_temperature)
            else:
                codes = self.sampling_hierarchy_parallel(hs, top_k, top_p, softmax_temperature, None)

            codes_level = []
            codes_level.append(codes[:, 0:1])
            if self.code_level > 1:
                codes_level.append(codes[:, 1:(1+4), ].unsqueeze(1))  # [B, 1, 2*2]
            if self.code_level > 2:
                codes_level.append(codes[:, (1+4):(1+4+16), ].unsqueeze(1))  # [B, 1, 4*4]

            return codes_level, presents

    def sampling_step_spatial(self,
                              sos: torch.FloatTensor,
                              codes: List[torch.LongTensor],
                              pos_codes_t: torch.LongTensor,
                              past: Optional[torch.Tensor] = None,
                              model_stage1: Optional[torch.nn.Module] = None
                              ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        B = sos.size(0)

        if codes[0] is None:
            assert past is None
            xs = sos
        else:
            if self.position_embedding == '1d':
                pos_emb = self.pos_emb_top(pos_codes_t)
                Ttop = self.pos_emb_top.weight.size(0)
                Htop = int(math.sqrt(Ttop))
            elif self.position_embedding == '2d':
                Htop = self.pos_emb_top_h.weight.size(0)
                pos_codes_h = pos_codes_t // Htop
                pos_codes_w = pos_codes_t % Htop
                pos_emb_h = self.pos_emb_top_h(pos_codes_h)
                pos_emb_w = self.pos_emb_top_w(pos_codes_w)
                pos_emb = pos_emb_h + pos_emb_w

            if self.spatial_embedding == 'transformer':
                emb_level0 = self.tok_emb_levels[0](codes[0])
                emb_level0 += pos_emb
                emb_level0 = rearrange(emb_level0, 'B L (U K) -> (B L) U K', U=1)
                hs = [emb_level0]

                emb_level1 = self.tok_emb_levels[1](codes[1])
                emb_level1 = rearrange(emb_level1, 'B L HW K -> (B L) HW K')
                hs.append(emb_level1)

                emb_level2 = self.tok_emb_levels[2](codes[2])
                emb_level2 = rearrange(emb_level2, 'B L HW K -> (B L) HW K')
                hs.append(emb_level2)

                xs = torch.cat(hs, dim=1)
                xps_embed = torch.arange(self.code_len, device=sos.device)
                xs += self.pos_emb_emb(xps_embed).unsqueeze(0)
                xs = self.emb_blocks(xs)
                xs = xs.mean(dim=1)
                xs = rearrange(xs, '(B L) K -> B L K', B=B)

            else:
                assert False

        xs = self.drop(xs)

        past = torch.cat(past, dim=-2) if past is not None else past
        presents = []
        for i, block in enumerate(self.blocks):
            xs, present = block.sample(xs, layer_past=None if past is None else past[i])
            presents.append(present)

        xs = self.ln_f(xs)

        return xs, presents

    def sampling_step_hierarchy_parallel(self,
                                         cnt: int,
                                         hs: torch.FloatTensor,
                                         codes: torch.LongTensor,
                                         pos_codes: torch.LongTensor,
                                         past: Optional[torch.Tensor] = None,
                                         model_stage1: Optional[torch.nn.Module] = None
                                         ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:

        B, L, K = hs.size()

        # the first step for level 0
        # output code     [Top0]
        # input embedding [h+sos]
        if codes is None:
            assert past is None
            if hs.size(1) > 1:
                hs = hs[:, self.idx_pred-1:self.idx_pred, :]
            xs = hs + self.sos_depth

        else:
            if 'add' in self.decoding_type and cnt == 2:
                # codes = [top, mid0, mid1, mid2, mid3]
                xs = self.tok_emb_depth_levels[cnt-1](codes[:, 1:])
            else:
                xs = self.tok_emb_depth_levels[cnt-1](codes)

            if cnt == 1:
                if 'reduce' in self.decoding_type:
                    xs = rearrange(xs, 'B HW1 (H2 W2 K) -> B (HW1 H2 W2) K', H2=2, W2=2)
                
                xs = xs + self.pos_emb_depths[cnt-1](pos_codes)
            elif cnt == 2:
                if 'reduce' in self.decoding_type:
                    xs = rearrange(xs, 'B (H1 W1) (H2 W2 K) -> B (H1 W1) (H2 W2) K', H1=2, W1=2, H2=2, W2=2)
                else:
                    xs = rearrange(xs, 'B (H1 W1 H2 W2) K -> B (H1 W1) (H2 W2) K', H1=2, W1=2, H2=1, W2=1)
                xs = xs + self.pos_emb_depths[cnt-1](pos_codes)

                # flattening
                if 'parallel' in self.decoding_type:
                    xs = rearrange(xs, 'B (H1 W1) (H2 W2) K -> B (H1 H2 W1 W2) K', B=B, H1=2, W1=2, H2=2, W2=2)
                else:
                    assert False

                if 'add' in self.decoding_type:
                    # add top code embedding
                    xs = xs + self.tok_emb_depth_levels[0](codes[:, 0:1])

        xs = self.drop(xs)

        past = torch.cat(past, dim=-2) if past is not None else past

        presents = []

        for i, block in enumerate(self.depths):
            xs, present = block.sample(xs, layer_past=None if past is None else past[i])
            presents.append(present)
        


        logits = self.head_levels[cnt](self.ln_levels[cnt](xs))
        if cnt == 0:
            logits = rearrange(logits, 'B HW K -> HW B K')
        elif cnt == 1:
            logits = rearrange(logits, 'B (H1 W1) K -> (H1 W1) B K', H1=2, W1=2)
        elif cnt == 2:
            if 'parallel' in self.decoding_type:
                logits = rearrange(logits, 'B (H1 H2 W1 W2) K -> (H1 H2 W1 W2) B K', H1=2, H2=2, W1=2, W2=2)

        return logits, presents

    @torch.no_grad()
    def sampling_hierarchy_parallel(self,
                                    hs: torch.FloatTensor,
                                    top_k: Optional[List[float]] = None,
                                    top_p: Optional[List[float]] = None,
                                    softmax_temperature: List[float] = [1.0, 1.0, 1.0],
                                    model_stage1: Optional[torch.nn.Module] = None,
                                    ) -> torch.LongTensor:
        B, L, K = hs.size()
        code = None
        past = None
        max_seq_len = self.code_level

        for cnt, h in enumerate(range(max_seq_len)):
            if code is None:
                code_ = None
                pos_enc_code_ = None
            else:
                code_ = code.clone().detach()

                if cnt == 1:
                    code_ = code_[:, 0:1]  # level 0 code
                    pos_enc_code_ = torch.arange(4, device=hs.device).repeat((B*L, 1))
                elif cnt == 2:
                    len_code = code_.size(1)
                    if 'add' in self.decoding_type:
                        code_ = code_[:, len_code-5:len_code]  # level 0 + 1 code
                    else:
                        code_ = code_[:, len_code-4:len_code]  # level 1 code

                    if 'tree' in self.decoding_type:
                        pos_enc_code_ = torch.arange(4, device=hs.device).repeat((B*L*4, 1))
                    elif 'parallel' in self.decoding_type:
                        pos_enc_code_ = torch.arange(16, device=hs.device).repeat((B*L, 1))
                        reorder_into_pyramid = 'BHW (H1 H2 W1 W2) -> BHW (H1 W1) (H2 W2)'
                        pos_enc_code_ = rearrange(pos_enc_code_, reorder_into_pyramid, H1=2, H2=2, W1=2, W2=2)

            _logits, present = self.sampling_step_hierarchy_parallel(cnt=cnt,
                                                                     hs=hs,
                                                                     codes=code_,
                                                                     pos_codes=pos_enc_code_,
                                                                     past=past,
                                                                     model_stage1=model_stage1)

            present = torch.stack(present).clone().detach()
            if past is None:
                past = [present]
            else:
                past.append(present)

            # iterate over local code dimension
            # iteration top code = 1, middle = 4, bottom = 16

            for logits in _logits:
                logits /= softmax_temperature[cnt]
                logits = cutoff_topk_logits(logits, top_k[cnt])
                probs = F.softmax(logits, dim=-1)
                probs = cutoff_topp_probs(probs, top_p[cnt])
                idx = torch.multinomial(probs, num_samples=1).clone().detach()
                code = idx if code is None else torch.cat([code, idx], axis=1)

        del past

        return code

    def forward_causal(self,
                       h: torch.FloatTensor,
                       codes: Tuple[torch.LongTensor],
                       model_stage1: Optional[torch.nn.Module] = None):

        top_codes, bot_codes = codes[0], codes[-1]

        B, Ttop = top_codes.shape
        B, Tbot = bot_codes.shape
        Htop = int(math.sqrt(Ttop))

        emb_level0 = self.tok_emb_depth_levels[0](codes[0])
        emb_level1 = self.tok_emb_depth_levels[1](codes[1])
        emb_level2 = self.tok_emb_depth_levels[2](codes[2])

        sos_depth = self.sos_depth.repeat((B*Ttop, 1, 1))

        xps_depth = torch.arange(1+4+16-1, device=top_codes.device).repeat((B*Ttop, 1))
        pos_depth = self.pos_emb_depths[0](xps_depth)

        # output code [Top0,  Mid0,      Mid1,      Mid2,      Mid3,      Bot0,     Bot1, ...     Bot15]
        # input code  [sos+h, Top0+PM0, Top0+PM1, Top0+PM2, Top0+PM3, Mid0+PB0, Mid0+PB1, ... Mid3+PB15]
        if self.use_txt_cond:
            h_txt = h[:, :self.idx_pred-1, :]
            logits_txt = self.head_txt(self.ln_txt(h_txt))

            h = h[:, self.idx_pred-1:, :]

        h = rearrange(h, 'B HW (U K) -> (B HW) U K', U=1)

        global_to_pyramid = 'B (H H1 H2 W W1 W2) K -> (B H W) (H1 W1) (H2 W2) K'
        emb_level0 = rearrange(emb_level0, global_to_pyramid, H1=1, H2=1, W1=1, W2=1, H=Htop, W=Htop)
        emb_level1 = rearrange(emb_level1, global_to_pyramid, H1=2, H2=2, W1=1, W2=1, H=Htop, W=Htop)
        emb_level2 = rearrange(emb_level2, global_to_pyramid, H1=2, H2=2, W1=2, W2=2, H=Htop, W=Htop)

        if 'add' in self.decoding_type:
            emb_level2 = (emb_level2 + emb_level1) + emb_level0
            emb_level1 = emb_level1 + emb_level0

        pyramid_to_flatten = 'BHW (H1 W1) (H2 W2) K -> BHW (H1 H2 W1 W2) K'
        emb_level0 = rearrange(emb_level0, pyramid_to_flatten, H1=1, H2=1, W1=1, W2=1)
        emb_level1 = rearrange(emb_level1, pyramid_to_flatten, H1=2, H2=2, W1=1, W2=1)
        emb_level2 = rearrange(emb_level2, pyramid_to_flatten, H1=2, H2=2, W1=2, W2=2)

        h = torch.cat([h, emb_level0, emb_level1, emb_level2[:, :-1, :]], dim=1)

        h_depth = torch.cat([sos_depth, pos_depth], dim=1)
        h = h + h_depth

        h = self.depths(h)

        logits_level0 = self.head_levels[0](self.ln_levels[0](h[:, 0, :]))
        logits_level1 = self.head_levels[1](self.ln_levels[1](h[:, 1:(1+4), :]))
        logits_level2 = self.head_levels[2](self.ln_levels[2](h[:, (1+4):(1+4+16), :]))

        logits = []
        logits.append(rearrange(logits_level0, '(B HW) K -> B HW K', B=B))
        logits.append(rearrange(logits_level1, '(B H W) (H1 W1) K -> B (H H1 W W1) K', H1=2, W1=2, H=Htop, W=Htop))
        pyramid_to_global = '(B H W) (H1 H2 W1 W2) K -> B (H H1 H2 W W1 W2) K'
        logits.append(rearrange(logits_level2, pyramid_to_global, H1=2, W1=2, H2=2, W2=2, H=Htop, W=Htop))

        if self.use_txt_cond:
            logits.append(logits_txt)

        return logits

    def sampling_step_depth_causal(self,
                                   hs: torch.FloatTensor,
                                   codes: torch.LongTensor,
                                   pos_codes: torch.LongTensor,
                                   cnt: int,
                                   past: Optional[torch.Tensor] = None
                                   ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:

        B, L, K = hs.size()

        # the first step
        if codes is None:
            assert past is None
            xs = hs + self.sos_depth.repeat((B, 1, 1))
            xs = self.drop(xs)

            presents = []
            for i, block in enumerate(self.depths):
                xs, present = block.sample(xs, layer_past=None)
                presents.append(present)

            level_cnt = 0

        # the other steps
        else:
            if pos_codes[0] == 0:
                xs = self.tok_emb_levels[0](codes)
            elif cnt < 5:
                xs = self.tok_emb_levels[1](codes)
            else:
                xs = self.tok_emb_levels[2](codes)

            xs = xs + self.pos_emb_depths[0](pos_codes)
            xs = self.drop(xs)

            past = torch.cat(past, dim=-2) if past is not None else past

            presents = []
            for i, block in enumerate(self.depths):
                xs, present = block.sample(xs, layer_past=None if past is None else past[i])
                presents.append(present)

            if cnt < 5:
                level_cnt = 1
            else:
                level_cnt = 2

        logits = self.head_levels[level_cnt](self.ln_levels[level_cnt](xs))
        return logits.squeeze(), presents

    @torch.no_grad()
    def sampling_depth_causal(self,
                              hs: torch.FloatTensor,
                              top_k: Optional[List[float]] = None,
                              top_p: Optional[List[float]] = None,
                              softmax_temperature: List[float] = 1.0) -> torch.LongTensor:
        code = None
        past = None
        max_seq_len = self.code_len

        for cnt, h in enumerate(range(max_seq_len)):
            if code is None:
                code_ = None
                pos_enc_code_ = None
            else:
                code_ = code.clone().detach()
                pos_enc_code_ = get_positional_encoding(code_, mode='1d')
                code_ = code_[:, cnt-1:cnt]
                pos_enc_code_ = pos_enc_code_[:, cnt-1:cnt]

            logits, present = self.sampling_step_depth_causal(hs=hs,
                                                              codes=code_,
                                                              pos_codes=pos_enc_code_,
                                                              cnt=cnt,
                                                              past=past)

            present = torch.stack(present).clone().detach()
            if past is None:
                past = [present]
            else:
                past.append(present)

            if cnt == 0:
                level_cnt = 0
            elif cnt < 5:
                level_cnt = 1
            else:
                level_cnt = 2

            _top_k = top_k[level_cnt]
            _top_p = top_p[level_cnt]

            logits /= softmax_temperature[level_cnt]

            logits = cutoff_topk_logits(logits, _top_k)
            probs = F.softmax(logits, dim=-1)
            probs = cutoff_topp_probs(probs, _top_p)

            idx = torch.multinomial(probs, num_samples=1).clone().detach()
            code = idx if code is None else torch.cat([code, idx], axis=1)

        del past
        return code

    def from_ckpt(self, path: str, strict: bool = True, ignore_keys: Optional[List] = None) -> None:
        ckpt = torch.load(path, map_location='cpu')['state_dict']
        if ignore_keys:
            for k in ignore_keys:
                del ckpt[k]
        self.load_state_dict(ckpt, strict=strict)
        print(f'{path} successfully restored..')
