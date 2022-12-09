# ------------------------------------------------------------------------------------
# HQ-Transformer for Two-level Modeling
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import math
import random
import copy
import torch
import torch.nn as nn

from typing import Optional, Tuple, List
from einops import rearrange
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.nn import functional as F

from hqvae.models.stage2.layers import Block, ParallelBlock
from hqvae.utils.sampling import cutoff_topk_logits, cutoff_topp_probs, get_positional_encoding


class iHQGPT(nn.Module):
    def __init__(self,
                 vocab_size_top: int,
                 vocab_size_bot: int,
                 vocab_size_txt: int,
                 ratio_bot2top: int,
                 use_cls_cond: bool,
                 use_txt_cond: bool,
                 model_type: str,
                 hparams: OmegaConf,
                 hparams_dec: OmegaConf = None) -> None:
        super().__init__()
        self.use_cls_cond = use_cls_cond
        self.use_txt_cond = use_txt_cond

        ##########################
        # transformer for top
        ##########################
        if 'parallel' in model_type:
            parallel_len = model_type.split('parallel')[-1]
            if len(parallel_len) > 0:
                self.bot_win = int(math.sqrt(int(parallel_len)))
            else:
                self.bot_win = 2
            model_type = 'parallel'
        elif 'bidirectional' in model_type:
            parallel_len = model_type.split('bidirectional')[-1]
            if len(parallel_len) > 0:
                self.bot_win = int(math.sqrt(int(parallel_len)))
            else:
                self.bot_win = 2
            model_type = 'bidirectional'
        else:
            self.bot_win = 1

        self.num_bottom_pred = self.bot_win * self.bot_win
        self.ratio_bot2top = ratio_bot2top
        self.len_seq_depth = 1 + self.ratio_bot2top // self.num_bottom_pred  # 1x1 top + 2x2 bottom codes
        self.top_win = int(math.sqrt(ratio_bot2top)) // self.bot_win

        # sos token embedding
        if self.use_cls_cond:
            self.sos = nn.Embedding(hparams.n_classes, hparams.embed_dim)
            self.idx_pred = 0
        elif self.use_txt_cond:
            self.tok_emb_txt = nn.Embedding(vocab_size_txt, hparams.embed_dim)
            self.pos_emb_txt = nn.Embedding(hparams.ctx_len_txt, hparams.embed_dim)

            self.head_txt = nn.Linear(hparams.embed_dim, vocab_size_txt, bias=False)
            self.ln_txt = nn.LayerNorm(hparams.embed_dim)

            self.idx_pred = hparams.ctx_len_txt

        else:
            self.sos = nn.Parameter(torch.randn(1, 1, hparams.embed_dim))
            self.idx_pred = 0

        # input embedding
        self.spatial_embedding = hparams.embedding_type
        pos_emb_dim = hparams.embed_dim
        if (hparams.embedding_type == 'reduce'):
            self.tok_emb_top = nn.Embedding(vocab_size_top, hparams.embed_dim)
            self.tok_emb_bot = nn.Embedding(vocab_size_bot, hparams.embed_dim//(self.ratio_bot2top))
            pos_emb_dim = hparams.embed_dim

        elif (hparams.embedding_type == 'multiple'):
            self.tok_emb_top = nn.Embedding(vocab_size_top, hparams.embed_dim)
            self.tok_emb_bot = nn.Embedding(vocab_size_bot, hparams.embed_dim)
            pos_emb_dim = hparams.embed_dim
            self.pos_emb_bot = nn.Parameter(torch.randn(1, 1, pos_emb_dim, self.num_bottom_pred))

        elif ('transformer' in hparams.embedding_type or 'bidirectional' in hparams.embedding_type):
            if 'transformer' in hparams.embedding_type:
                tok = 'transformer'
            elif 'bidirectional' in hparams.embedding_type:
                tok = 'bidirectional'
            self.spatial_embedding = tok
            n_layers_emb = int(hparams.embedding_type.split(tok)[-1])
            self.tok_emb_top = nn.Embedding(vocab_size_top, hparams.embed_dim)
            self.tok_emb_bot = nn.Embedding(vocab_size_bot, hparams.embed_dim)
            self.pos_emb_emb = nn.Embedding(self.ratio_bot2top+1, pos_emb_dim)
            self.emb_blocks = [Block(ctx_len=self.ratio_bot2top + 1,
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
            assert (False)

        self.position_embedding = hparams.position_embedding
        if (self.position_embedding == '1d'):
            self.pos_emb_top = nn.Embedding(hparams.ctx_len_img, pos_emb_dim)
        elif (self.position_embedding == '2d'):
            H = int(math.sqrt(hparams.ctx_len_img))
            self.pos_emb_top_h = nn.Embedding(H, pos_emb_dim)
            self.pos_emb_top_w = nn.Embedding(H, pos_emb_dim)

        self.use_random_order = hparams.use_random_order
        self.rate_random_order = hparams.rate_random_order
        if (self.use_random_order):
            self.pred_emb_top = nn.Embedding(hparams.ctx_len_img, hparams.embed_dim)

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

        ###############################
        # depth transformer for bottom
        ###############################

        if hparams_dec is None:
            print('hparam_dec is None. Use hparam instead.')
            hparams_dec = copy.deepcopy(hparams)
            hparams_dec.n_layers = 4  # same with depth trans. in RQ

        # sos token embedding
        self.sos_depth = nn.Parameter(torch.randn(1, 1, hparams_dec.embed_dim))

        # input embedding
        self.depth_embedding = 'baseline'
        self.tok_emb_top_depth = nn.Embedding(vocab_size_top, hparams_dec.embed_dim)

        if (self.len_seq_depth > 2):
            self.tok_emb_bot_depth = nn.Embedding(vocab_size_bot, hparams_dec.embed_dim)
        else:
            self.tok_emb_bot_depth = nn.Embedding(vocab_size_bot, hparams_dec.embed_dim)

        self.pos_emb_depth = nn.Embedding(max(self.len_seq_depth, 5), hparams_dec.embed_dim)
        if ('parallel' == model_type):
            if (self.ratio_bot2top == 16):
                self.pos_emb_depth = nn.Embedding(16, hparams_dec.embed_dim)

        # depth transformer blocks
        if ('parallel' == model_type):
            self.depths = [ParallelBlock(ctx_len=self.len_seq_depth+1,
                                         embed_dim=hparams_dec.embed_dim,
                                         n_heads=hparams_dec.n_heads,
                                         mlp_bias=hparams_dec.mlp_bias,
                                         attn_bias=hparams_dec.attn_bias,
                                         resid_pdrop=hparams_dec.resid_pdrop,
                                         attn_pdrop=hparams_dec.attn_pdrop,
                                         gelu_use_approx=hparams_dec.gelu_use_approx,
                                         parallel_len=self.num_bottom_pred) for i in range(1, hparams_dec.n_layers+1)]
        elif ('bidirectional' == model_type):
            self.depths = [Block(ctx_len=self.len_seq_depth+1,
                                 embed_dim=hparams_dec.embed_dim,
                                 n_heads=hparams_dec.n_heads,
                                 mlp_bias=hparams_dec.mlp_bias,
                                 attn_bias=hparams_dec.attn_bias,
                                 resid_pdrop=hparams_dec.resid_pdrop,
                                 attn_pdrop=hparams_dec.attn_pdrop,
                                 gelu_use_approx=hparams_dec.gelu_use_approx,
                                 causal_attn=False) for i in range(1, hparams_dec.n_layers+1)]
        else:
            self.depths = [Block(ctx_len=self.len_seq_depth+1,
                                 embed_dim=hparams_dec.embed_dim,
                                 n_heads=hparams_dec.n_heads,
                                 mlp_bias=hparams_dec.mlp_bias,
                                 attn_bias=hparams_dec.attn_bias,
                                 resid_pdrop=hparams_dec.resid_pdrop,
                                 attn_pdrop=hparams_dec.attn_pdrop,
                                 gelu_use_approx=hparams_dec.gelu_use_approx) for i in range(1, hparams_dec.n_layers+1)]
        self.depths = nn.Sequential(*self.depths)

        # head
        self.ln_top = nn.LayerNorm(hparams_dec.embed_dim)
        self.head_top = nn.Linear(hparams_dec.embed_dim, vocab_size_top, bias=False)

        self.ln_bot = nn.LayerNorm(hparams_dec.embed_dim)
        self.head_bot = nn.Linear(hparams_dec.embed_dim, vocab_size_bot, bias=False)

        self.ctx_len_img = hparams.ctx_len_img
        self.n_layers = hparams.n_layers
        self.n_layers_depth = hparams_dec.n_layers
        self.model_type = model_type

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_random_order(self, top_code):
        B = top_code.size(0)
        if (self.use_random_order):
            if (self.training and random.random() < self.rate_random_order):
                # raster scan order -> [order] -> randomly-permuted order
                order = torch.stack([torch.randperm(64, device=top_code.device) for i in range(0, B)])
                # randomly-permuted order -> [unorder] -> raster scan order
                unorder = torch.argsort(order, dim=1)
            else:
                # use raster order for validation
                order = torch.stack([torch.arange(0, 64, device=top_code.device) for i in range(0, B)])
                unorder = torch.stack([torch.arange(63, -1, -1, device=top_code.device) for i in range(0, B)])

            randperm = {'order': order, 'unorder': unorder}
        else:
            randperm = None

        return randperm

    def forward(self,
                codes: Tuple[torch.LongTensor],
                labels: Optional[torch.LongTensor] = None,
                model_stage1: Optional[torch.nn.Module] = None) -> torch.FloatTensor:

        order = self.get_random_order(codes[0])

        # main transformer
        h = self.forward_main(codes, labels, model_stage1, order)

        # depth transformer
        return self.forward_depth(h, codes, model_stage1, order)

    def forward_main(self,
                     codes: Tuple[torch.LongTensor],
                     labels: Optional[torch.LongTensor] = None,
                     model_stage1: Optional[torch.nn.Module] = None,
                     order: Optional[torch.LongTensor] = None):
        top_codes, bot_codes = codes[0], codes[1]

        B, Ttop = top_codes.shape
        B, Tbot = bot_codes.shape
        Htop = int(math.sqrt(Ttop))
        Hbw, Htw = self.bot_win, self.top_win

        # main transformer
        # single token = single top code [1] + top_pos +  four bot codes [1 3]
        #                                                                [2 4]
        if (self.position_embedding == '1d'):
            xps = torch.arange(Ttop, device=top_codes.device).repeat((B, 1))
            pos_emb = self.pos_emb_top(xps)

        elif (self.position_embedding == '2d'):
            xs_pos_h = torch.arange(Htop, device=top_codes.device).repeat(B, Htop, 1).transpose(1, 2)
            xs_pos_w = torch.arange(Htop, device=top_codes.device).repeat(B, Htop, 1)
            pos_emb_h = self.pos_emb_top_h(xs_pos_h)
            pos_emb_w = self.pos_emb_top_w(xs_pos_w)
            pos_emb = pos_emb_h + pos_emb_w
            pos_emb = rearrange(pos_emb, 'B H W C -> B (H W) C')

        if (self.spatial_embedding == 'reduce'):  # 1/4 channel size of baseline and without projection
            h_top = self.tok_emb_top(top_codes) + pos_emb
            h_bot = self.tok_emb_bot(bot_codes)
            h_bot = rearrange(h_bot, 'B (H H2 W W2) K -> B (H W) (K H2 W2)',
                              H2=self.top_win * self.bot_win, W2=self.top_win * self.bot_win, H=Htop, W=Htop)
            h = h_top + h_bot

        elif (self.spatial_embedding == 'multiple'):
            h_top = self.tok_emb_top(top_codes) + pos_emb
            h_bot = self.tok_emb_bot(bot_codes)
            h_bot = rearrange(h_bot, 'B (H H2 W W2) K -> B (H W) K (H2 W2)',
                              H2=self.top_win * self.bot_win, W2=self.top_win * self.bot_win, H=Htop, W=Htop)
            h = h_top + (h_bot * self.pos_emb_bot).sum(-1)
        elif (self.spatial_embedding == 'transformer' or self.spatial_embedding == 'bidirectional'):
            emb_top = self.tok_emb_top(top_codes)
            if self.spatial_embedding == 'transformer':
                emb_top += pos_emb

            emb_bot = self.tok_emb_bot(bot_codes)
            emb_top = rearrange(emb_top, 'B L (U K) -> (B L) U K', U=1)
            emb_bot = rearrange(emb_bot, 'B (H H2 W W2) K -> (B H W) (H2 W2) K', H2=Htw*Hbw, W2=Htw*Hbw, H=Htop, W=Htop)
            h = torch.cat([emb_top, emb_bot], dim=1)
            xps_embed = torch.arange(self.ratio_bot2top+1, device=top_codes.device)
            h += self.pos_emb_emb(xps_embed).unsqueeze(0)
            h = self.emb_blocks(h)
            h = h.mean(dim=1)
            h = rearrange(h, '(B L) K -> B L K', B=B)

            if self.spatial_embedding == 'bidirectional':
                h += pos_emb
        else:
            assert (False)

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

    def forward_depth(self,
                      h: torch.FloatTensor,
                      codes: Tuple[torch.LongTensor],
                      model_stage1: Optional[torch.nn.Module] = None,
                      order: Optional[torch.LongTensor] = None):

        top_codes, bot_codes = codes[0], codes[1]

        B, Ttop = top_codes.shape
        B, Tbot = bot_codes.shape
        Htop = int(math.sqrt(Ttop))
        Hbw, Htw = self.bot_win, self.top_win

        sos_depth = self.sos_depth.repeat((B*Ttop, 1, 1))

        emb_top = self.tok_emb_top_depth(top_codes)

        if (self.model_type == 'top2bot'): # causal transformer
            emb_bot = self.tok_emb_bot_depth(bot_codes)

            xps_top_depth = torch.arange(1, device=top_codes.device).repeat((B*Ttop, 1))
            xps_bot_depth = torch.arange(1, self.len_seq_depth, device=bot_codes.device).repeat((B*Ttop, 1))

            pos_top_depth = self.pos_emb_depth(xps_top_depth)
            pos_bot_depth = self.pos_emb_depth(xps_bot_depth)

            # output code [Top0,  Bot0, Bot1, Bot2, Bot3]
            # input code  [sos+h, Top0, Bot0, Bot1, Bot2]

            h = rearrange(h, 'B L (U K) -> (B L) U K', U=1) + sos_depth
            emb_top = rearrange(emb_top, 'B L (U K) -> (B L) U K', U=1) + pos_top_depth
            emb_bot = rearrange(emb_bot, 'B (H H2 W W2) K -> (B H W) (H2 W2) K', H2=Htw*Hbw, W2=Htw*Hbw, H=Htop, W=Htop)
            emb_bot = emb_bot + pos_bot_depth

            h = torch.cat([h, emb_top, emb_bot[:, 0:self.len_seq_depth-2, :]], dim=1)
            h = self.depths(h)

            logits_top = self.head_top(self.ln_top(h[:, 0, :]))
            logits_bot = self.head_bot(self.ln_bot(h[:, 1:, :]))

            logits_top = rearrange(logits_top, '(B L) K -> B L K', B=B)
            logits_bot = rearrange(logits_bot, '(B H W) (H2 W2) K -> B (H H2 W W2) K',
                                   H2=Htw*Hbw, W2=Htw*Hbw, H=Htop, W=Htop)

        elif (self.model_type == 'parallel'):
            xps_top_depth = torch.arange(self.ratio_bot2top, device=top_codes.device).repeat((B*Ttop, 1))
            pos_top_depth = self.pos_emb_depth(xps_top_depth)

            # output code [Top0,  Bot0,      Bot1,      Bot2,      Bot3]
            # input code  [sos+h, Top0+Pos0, Top0+Pos1, Top0+Pos2, Top0+Pos3]
            if self.use_txt_cond:
                h_txt = h[:, :self.idx_pred-1, :]
                logits_txt = self.head_txt(self.ln_txt(h_txt))

                h = h[:, self.idx_pred-1:, :]

            h = rearrange(h, 'B L (U K) -> (B L) U K', U=1) + sos_depth
            emb_top = rearrange(emb_top, 'B L (U K) -> (B L) U K', U=1) + pos_top_depth

            h = torch.cat([h, emb_top], dim=1)
            h = self.depths(h)

            logits_top = self.head_top(self.ln_top(h[:, 0, :]))
            logits_bot = self.head_bot(self.ln_bot(h[:, 1:, :]))

            logits_top = rearrange(logits_top, '(B L) K -> B L K', B=B)
            pyramid_to_global = '(B H W) (H2 W2) K -> B (H H2 W W2) K'
            logits_bot = rearrange(logits_bot, pyramid_to_global, H2=Htw*Hbw, W2=Htw*Hbw, H=Htop, W=Htop)

        elif (self.model_type == 'bidirectional'):
            xps_top_depth = torch.arange(self.ratio_bot2top, device=top_codes.device).repeat((B*Ttop, 1))
            pos_top_depth = self.pos_emb_depth(xps_top_depth)

            # output code [Top0,  Bot0,      Bot1,      Bot2,      Bot3]
            # input code  [sos+h, Top0+Pos0, Top0+Pos1, Top0+Pos2, Top0+Pos3]

            h = rearrange(h, 'B L (U K) -> (B L) U K', U=1) + sos_depth
            h = torch.cat([h, pos_top_depth], dim=1)
            h = self.depths(h)

            logits_top = self.head_top(self.ln_top(h[:, 0, :]))
            logits_bot = self.head_bot(self.ln_bot(h[:, 1:, :]))

            logits_top = rearrange(logits_top, '(B L) K -> B L K', B=B)
            pyramid_to_global = '(B H W) (H2 W2) K -> B (H H2 W W2) K'
            logits_bot = rearrange(logits_bot, pyramid_to_global, H2=Htw*Hbw, W2=Htw*Hbw, H=Htop, W=Htop)

        if self.use_txt_cond:
            return (logits_top, logits_bot, logits_txt)
        else:
            return (logits_top, logits_bot)

    @torch.no_grad()
    def sampling_step(self,
                      sos: torch.FloatTensor,
                      codes_t: torch.LongTensor,
                      codes_b: torch.LongTensor,
                      pos_codes: torch.LongTensor,
                      use_fp16: bool = True,
                      top_k_top: Optional[float] = None,
                      top_p_top: Optional[float] = None,
                      top_k_bot: Optional[float] = None,
                      top_p_bot: Optional[float] = None,
                      softmax_temperature: List[float] = [1.0, 1.0],
                      past: Optional[torch.Tensor] = None,
                      model_stage1: Optional[torch.nn.Module] = None,
                      given_top_code: Optional[torch.LongTensor] = None
                      ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:

        with autocast(enabled=use_fp16):
            hs, presents = self.sampling_step_spatial(sos, codes_t, codes_b, pos_codes, past, model_stage1)

            if (self.model_type == 'top2bot'):
                codes = self.sampling_depth_baseline(hs,
                                                     top_k_top,
                                                     top_p_top,
                                                     top_k_bot,
                                                     top_p_bot,
                                                     softmax_temperature)
                code_top = codes[:, 0:1]
                code_bot = codes[:, 1:, ].unsqueeze(1)  # [B, 1, KH*KW]

            elif (self.model_type == 'parallel'):
                codes = self.sampling_depth_parallel(hs,
                                                     top_k_top,
                                                     top_p_top,
                                                     top_k_bot,
                                                     top_p_bot,
                                                     softmax_temperature,
                                                     None,
                                                     given_top_code)
                code_top = codes[:, 0:1]
                code_bot = codes[:, 1:, ].unsqueeze(1)  # [B, 1, KH*KW]

            elif (self.model_type == 'bidirectional'):
                codes = self.sampling_depth_bidirectional(hs,
                                                          top_k_top,
                                                          top_p_top,
                                                          top_k_bot,
                                                          top_p_bot,
                                                          softmax_temperature)
                code_top = codes[:, 0:1]
                code_bot = codes[:, 1:, ].unsqueeze(1)  # [B, 1, KH*KW]

            return code_top, code_bot, presents

    def sampling_step_spatial(self,
                              sos: torch.FloatTensor,
                              codes_t: torch.LongTensor,
                              codes_b: torch.LongTensor,
                              pos_codes_t: torch.LongTensor,
                              past: Optional[torch.Tensor] = None,
                              model_stage1: Optional[torch.nn.Module] = None
                              ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        B = sos.size(0)
        Hbw, Htw = self.bot_win, self.top_win

        if codes_t is None:
            assert past is None
            if (self.use_random_order):
                xs = sos + self.pred_emb_top(torch.zeros(B, device=sos.device, dtype=int))
            else:
                xs = sos
            xs = self.drop(sos)
            presents = []
            for i, block in enumerate(self.blocks):
                xs, present = block.sample(xs, layer_past=None)
                presents.append(present)
            xs = self.ln_f(xs)
        else:
            if (self.position_embedding == '1d'):
                pos_emb = self.pos_emb_top(pos_codes_t)
            elif (self.position_embedding == '2d'):
                Htop = self.pos_emb_top_h.weight.size(0)
                pos_codes_h = pos_codes_t // Htop
                pos_codes_w = pos_codes_t % Htop
                pos_emb_h = self.pos_emb_top_h(pos_codes_h)
                pos_emb_w = self.pos_emb_top_w(pos_codes_w)
                pos_emb = pos_emb_h + pos_emb_w

            if (self.spatial_embedding == 'baseline'):
                xs_top = self.tok_emb_top(codes_t) + pos_emb
                xs_bot = self.tok_emb_bot(codes_b)
                xs_bot = rearrange(xs_bot, 'B (U L) K -> B U (K L)', U=1)
                xs = self.embd_proj(xs_top + xs_bot)

            elif (self.spatial_embedding == 'reduce'):
                xs_top = self.tok_emb_top(codes_t) + pos_emb
                xs_bot = self.tok_emb_bot(codes_b)
                xs_bot = rearrange(xs_bot, 'B (U L) K -> B U (K L)', U=1)
                xs = xs_top + xs_bot

            elif (self.spatial_embedding == 'multiple'):
                xs_top = self.tok_emb_top(codes_t) + pos_emb
                xs_bot = self.tok_emb_bot(codes_b)
                xs_bot = rearrange(xs_bot, 'B (U L) K -> B U K L', U=1)
                xs = xs_top + (xs_bot * self.pos_emb_bot).sum(-1)

            elif (self.spatial_embedding == 'transformer'):
                emb_top = self.tok_emb_top(codes_t) + pos_emb
                emb_bot = self.tok_emb_bot(codes_b)
                emb_top = rearrange(emb_top, 'B L (U K) -> (B L) U K', U=1)
                emb_bot = rearrange(emb_bot, 'B (H H2 W W2) K -> (B H W) (H2 W2) K', H2=Htw*Hbw, W2=Htw*Hbw, H=1, W=1)
                h = torch.cat([emb_top, emb_bot], dim=1)
                xps_embed = torch.arange(self.ratio_bot2top+1, device=codes_t.device)
                h += self.pos_emb_emb(xps_embed).unsqueeze(0)
                h = self.emb_blocks(h)
                h = h.mean(dim=1)
                xs = rearrange(h, '(B L) K -> B L K', B=B)

            else:
                assert (False)

            if (self.use_random_order):
                xs = xs + self.pred_emb_top(pos_codes_t + 1)

            xs = self.drop(xs)

            past = torch.cat(past, dim=-2) if past is not None else past

            presents = []
            for i, block in enumerate(self.blocks):
                xs, present = block.sample(xs, layer_past=None if past is None else past[i])
                presents.append(present)

            xs = self.ln_f(xs)

        return xs, presents

    def sampling_step_depth_baseline(self,
                                     hs: torch.FloatTensor,
                                     codes: torch.LongTensor,
                                     pos_codes: torch.LongTensor,
                                     past: Optional[torch.Tensor] = None
                                     ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:

        B, L, K = hs.size()

        # top2bot
        # output code [Top0,  Bot0, Bot1, Bot2, Bot3]
        # input code  [sos+h, Top0, Bot0, Bot1, Bot2]

        # the first step
        if codes is None:
            assert past is None
            xs = hs + self.sos_depth.repeat((B, 1, 1))
            xs = self.drop(xs)

            presents = []
            for i, block in enumerate(self.depths):
                xs, present = block.sample(xs, layer_past=None)
                presents.append(present)

            xs = self.ln_top(xs)
            logits = self.head_top(xs)

        # the other steps
        else:
            if pos_codes[0] == 0:
                xs = self.tok_emb_top_depth(codes)
            else:
                xs = self.tok_emb_bot_depth(codes)
            xs = xs + self.pos_emb_depth(pos_codes)
            xs = self.drop(xs)

            past = torch.cat(past, dim=-2) if past is not None else past

            presents = []
            for i, block in enumerate(self.depths):
                xs, present = block.sample(xs, layer_past=None if past is None else past[i])
                presents.append(present)

            xs = self.ln_bot(xs)
            logits = self.head_bot(xs)

        return logits.squeeze(), presents

    @torch.no_grad()
    def sampling_depth_baseline(self,
                                hs: torch.FloatTensor,
                                top_k_top: Optional[float] = None,
                                top_p_top: Optional[float] = None,
                                top_k_bot: Optional[float] = None,
                                top_p_bot: Optional[float] = None,
                                softmax_temperatures: List[float] = [1.0, 1.0]) -> torch.LongTensor:
        code = None
        past = None
        max_seq_len = self.len_seq_depth

        for cnt, h in enumerate(range(max_seq_len)):
            if code is None:
                code_ = None
                pos_enc_code_ = None
            else:
                code_ = code.clone().detach()
                pos_enc_code_ = get_positional_encoding(code_, mode='1d')
                code_ = code_[:, cnt-1:cnt]
                pos_enc_code_ = pos_enc_code_[:, cnt-1:cnt]

            logits, present = self.sampling_step_depth_baseline(hs=hs,
                                                                codes=code_,
                                                                pos_codes=pos_enc_code_,
                                                                past=past)

            present = torch.stack(present).clone().detach()
            if past is None:
                past = [present]
            else:
                past.append(present)
            if (cnt == 0):
                top_k = top_k_top
                top_p = top_p_top
                softmax_temperature = softmax_temperatures[0]
            else:
                top_k = top_k_bot
                top_p = top_p_bot
                softmax_temperature = softmax_temperatures[1]

            logits /= softmax_temperature

            logits = cutoff_topk_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            probs = cutoff_topp_probs(probs, top_p)

            idx = torch.multinomial(probs, num_samples=1).clone().detach()
            code = idx if code is None else torch.cat([code, idx], axis=1)

        del past
        return code


    def sampling_step_depth_parallel(self,
                                     hs: torch.FloatTensor,
                                     codes: torch.LongTensor,
                                     pos_codes: torch.LongTensor,
                                     past: Optional[torch.Tensor] = None,
                                     model_stage1: Optional[torch.nn.Module] = None
                                     ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:

        B, L, K = hs.size()

        # top2parallel
        # output code [Top0,  Bot0,      Bot1,      Bot2,      Bot3]
        # input code  [sos+h, Top0+Pos0, Top0+Pos1, Top0+Pos2, Top0+Pos3]

        # the first step
        if codes is None:
            assert past is None
            if (hs.size(1) > 1):
                hs = hs[:, self.idx_pred-1:self.idx_pred, :]
            xs = hs + self.sos_depth.repeat((B, 1, 1))
            xs = self.drop(xs)

            presents = []
            for i, block in enumerate(self.depths):
                xs, present = block.sample(xs, layer_past=None)
                presents.append(present)

            xs = self.ln_top(xs)
            logits = self.head_top(xs)
        else:
            if 'embed' in self.depth_embedding:
                xs = model_stage1.quantize_t.get_codebook_entry(codes)
                xs = self.embed_proj_top_depth(xs.detach())
            else:
                xs = self.tok_emb_top_depth(codes)

            xs = xs + self.pos_emb_depth(pos_codes)
            xs = self.drop(xs)

            past = torch.cat(past, dim=-2) if past is not None else past

            presents = []

            for i, block in enumerate(self.depths):
                xs, present = block.sample(xs, layer_past=None if past is None else past[i])
                presents.append(present)

            xs = self.ln_bot(xs)
            logits = self.head_bot(xs)

            logits = rearrange(logits, 'B L K -> L B K')

        return logits.squeeze(), presents

    @torch.no_grad()
    def sampling_depth_parallel(self,
                                hs: torch.FloatTensor,
                                top_k_top: Optional[float] = None,
                                top_p_top: Optional[float] = None,
                                top_k_bot: Optional[float] = None,
                                top_p_bot: Optional[float] = None,
                                softmax_temperatures: List[float] = [1.0, 1.0],
                                model_stage1: Optional[torch.nn.Module] = None,
                                given_top_code: Optional[torch.LongTensor] = None,
                                ) -> torch.LongTensor:
        code = None
        past = None
        max_seq_len = self.len_seq_depth  # top (first), quadbot (second)

        for cnt, h in enumerate(range(max_seq_len)):
            if code is None:
                code_ = None
                pos_enc_code_ = None
            else:
                code_ = code.clone().detach()
                pos_enc_code_ = torch.arange(self.ratio_bot2top, device=code_.device).repeat((code.size(0), 1))
                if (cnt > 1):
                    len_code = code_.size(1)
                    code_ = code_[:, len_code-self.num_bottom_pred:len_code]
                    pos_enc_code_ = pos_enc_code_[:, self.num_bottom_pred*(cnt-1):self.num_bottom_pred*cnt]
                else:
                    code_ = code_[:, cnt-1:cnt]

            _logits, present = self.sampling_step_depth_parallel(hs=hs,
                                                                 codes=code_,
                                                                 pos_codes=pos_enc_code_,
                                                                 past=past,
                                                                 model_stage1=model_stage1)

            present = torch.stack(present).clone().detach()
            if past is None:
                past = [present]
            else:
                past.append(present)

            if (cnt == 0):
                _logits /= softmax_temperatures[0]
                logits = cutoff_topk_logits(_logits, top_k_top)
                probs = F.softmax(logits, dim=-1)
                probs = cutoff_topp_probs(probs, top_p_top)

                if (given_top_code is None):
                    idx = torch.multinomial(probs, num_samples=1).clone().detach()
                else:
                    if (hs.size(0) == given_top_code.size(0)):
                        idx = given_top_code
                    else:
                        idx = given_top_code.repeat(hs.size(0), 1)
                code = idx if code is None else torch.cat([code, idx], axis=1)
            else:
                # quadbottom has four bottom logits within a single step sampling
                for logits in _logits:
                    logits /= softmax_temperatures[1]
                    logits = cutoff_topk_logits(logits, top_k_bot)
                    probs = F.softmax(logits, dim=-1)
                    probs = cutoff_topp_probs(probs, top_p_bot)

                    idx = torch.multinomial(probs, num_samples=1).clone().detach()
                    code = idx if code is None else torch.cat([code, idx], axis=1)

        del past

        return code

    def sampling_step_depth_bidirectional(self,
                                          hs: torch.FloatTensor,
                                          codes: torch.LongTensor,
                                          pos_codes: torch.LongTensor,
                                          past: Optional[torch.Tensor] = None,
                                          model_stage1: Optional[torch.nn.Module] = None
                                          ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:

        B, L, K = hs.size()

        # top2parallel
        # output code [Top0,  Bot0, Bot1, Bot2, Bot3]
        # input code  [sos+h, Pos0, Pos1, Pos2, Pos3]

        # the first step
        if codes is None:
            assert past is None
            xs = hs + self.sos_depth.repeat((B, 1, 1))
            xs_pos = self.pos_emb_depth(pos_codes)
            xs = torch.cat([xs, xs_pos], dim=1)
            xs = self.drop(xs)

            presents = []
            for i, block in enumerate(self.depths):
                xs, present = block.sample(xs, layer_past=None)
                presents.append(present)

            logits_top = self.head_top(self.ln_top(xs[:, 0:1, :]))
            logits_bot = self.head_bot(self.ln_bot(xs[:, 1:, :]))
            logits = torch.cat([logits_top, logits_bot], dim=1)
            logits = rearrange(logits, 'B L K -> L B K')

        else:
            assert (False)

        return logits, presents

    @torch.no_grad()
    def sampling_depth_bidirectional(self,
                                     hs: torch.FloatTensor,
                                     top_k_top: Optional[float] = None,
                                     top_p_top: Optional[float] = None,
                                     top_k_bot: Optional[float] = None,
                                     top_p_bot: Optional[float] = None,
                                     softmax_temperatures: List[float] = [1.0, 1.0],
                                     model_stage1: Optional[torch.nn.Module] = None,
                                     ) -> torch.LongTensor:
        B = hs.size(0)
        code = None
        past = None
        max_seq_len = 1  # top (first), quadbot (second)

        for cnt, h in enumerate(range(max_seq_len)):
            if code is None:
                code_ = None
                pos_enc_code_ = torch.arange(self.ratio_bot2top, device=hs.device).repeat((B, 1))
            else:
                assert (False)
            _logits, present = self.sampling_step_depth_bidirectional(hs=hs,
                                                                      codes=code_,
                                                                      pos_codes=pos_enc_code_,
                                                                      past=past,
                                                                      model_stage1=model_stage1)

            present = torch.stack(present).clone().detach()
            if past is None:
                past = [present]
            else:
                past.append(present)

            if cnt == 0:
                softmax_temperature = softmax_temperatures[0]
            else:
                softmax_temperature = softmax_temperatures[1]

            # quadbottom has four bottom logits within a single step sampling
            for logits in _logits:
                logits /= softmax_temperature
                logits = cutoff_topk_logits(logits, top_k_bot)
                probs = F.softmax(logits, dim=-1)
                probs = cutoff_topp_probs(probs, top_p_bot)

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
