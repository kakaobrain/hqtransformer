# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional, Tuple
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.cuda.amp import autocast

from hqvae.utils.sampling import sampling_igpt
from hqvae.utils.config2 import get_base_config
from hqvae.optimizers.scheduler import build_scheduler

from .stage1.vqgan import VQGAN
from .stage1.generator import VQGANGenerator, VQGAN2Generator, SimRQGAN2Generator
from .stage1.generator import HQVAEGenerator
from .stage2.transformer import Transformer1d, iGPT
from .stage2.hierarchical_ar import iHQGPT
from .stage2.hqtransformer import HQTransformer


def log_prob_from_logits(x, axis=1):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering -> NCHW format
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True) + 1e-7)


def soft_target_cross_entropy(input, target, reduction='mean', label_smoothing=0.0):
    unif = torch.ones_like(target) / target.shape[-1]
    target = label_smoothing * unif + (1-label_smoothing) * target
    loss = torch.sum(-target * log_prob_from_logits(input, axis=-1), dim=-1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError()


def build_model(model_name, cfg_stage1, cfg_opt):
    if model_name == 'vqgan':
        generator = VQGANGenerator(n_embed=cfg_stage1.n_embed,
                                   embed_dim=cfg_stage1.embed_dim,
                                   ema_update=cfg_stage1.ema_update,
                                   hparams=cfg_stage1.hparams)
    elif model_name == 'vqgan2':
        generator = VQGAN2Generator(n_embed=cfg_stage1.n_embed,
                                    embed_dim=cfg_stage1.embed_dim,
                                    ema_update=cfg_stage1.ema_update,
                                    hparams=cfg_stage1.hparams,
                                    hparams_aux=cfg_stage1.hparams_aux)
    elif model_name == 'simrqgan2':
        generator = SimRQGAN2Generator(n_embed=cfg_stage1.n_embed,
                                       embed_dim=cfg_stage1.embed_dim,
                                       ema_update=cfg_stage1.ema_update,
                                       hparams=cfg_stage1.hparams,
                                       hparams_aux=cfg_stage1.hparams_aux)
    elif model_name == 'hqvae':
        if (hasattr(cfg_stage1, 'n_embed_levels')):
            n_embed_levels = cfg_stage1.n_embed_levels
        else:
            n_embed_levels = [cfg_stage1.n_embed for i in range(0, cfg_stage1.hparams_aux.code_levels)]

        generator = HQVAEGenerator(n_embed_levels=n_embed_levels,
                                   embed_dim=cfg_stage1.embed_dim,
                                   ema_update=cfg_stage1.ema_update,
                                   hparams=cfg_stage1.hparams,
                                   hparams_aux=cfg_stage1.hparams_aux)
    else:
        raise ValueError(f'{model_name} is not supported..')

    if hasattr(cfg_stage1, 'hparams_disc'):
        hparams_disc = cfg_stage1.hparams_disc
    else:
        hparams_disc = None

    return VQGAN(generator=generator,
                 hparams_disc=hparams_disc,
                 hparams_opt=cfg_opt)


class ImageGPT2(pl.LightningModule):
    def __init__(self,
                 config: OmegaConf) -> None:
        super().__init__()
        if (config.stage1.type == 'simrqgan2'):
            self.stage1 = SimRQGAN2Generator(n_embed=config.stage1.n_embed,
                                             embed_dim=config.stage1.embed_dim,
                                             ema_update=config.stage1.ema_update,
                                             hparams=config.stage1.hparams,
                                             hparams_aux=config.stage1.hparams_aux)
        elif (config.stage1.type == 'hqvae'):
            if (hasattr(config.stage1, 'n_embed_levels')):
                n_embed_levels = config.stage1.n_embed_levels
            else:
                n_embed_levels = [config.stage1.n_embed for i in range(0, config.stage1.hparams_aux.code_levels)]

            self.stage1 = HQVAEGenerator(n_embed_levels=n_embed_levels,
                                         embed_dim=config.stage1.embed_dim,
                                         ema_update=config.stage1.ema_update,
                                         hparams=config.stage1.hparams,
                                         hparams_aux=config.stage1.hparams_aux)

        if config.stage2.type == 'top':
            self.stage2 = iGPT(vocab_size_img=config.stage2.vocab_size_img,
                               use_cls_cond=config.stage2.use_cls_cond,
                               hparams=config.stage2.hparams)
        elif config.stage2.type == 'bottom':
            self.stage2 = Transformer1d(vocab_size_txt=config.stage2.vocab_size_img,
                                        vocab_size_img=config.stage2.vocab_size_img,
                                        hparams=config.stage2.hparams)

        elif 'hq-transformer' in config.stage2.type:
            if '/' in config.stage2.type:
                model_type = config.stage2.type.split('/')[-1]
            else:
                model_type = 'top2bot'  # baseline method

            self.stage2 = iHQGPT(vocab_size_top=config.stage2.vocab_size_img,
                                 vocab_size_bot=config.stage2.vocab_size_img,
                                 vocab_size_txt=config.stage2.vocab_size_txt,
                                 ratio_bot2top=config.stage2.ratio_bot2top,
                                 use_cls_cond=config.stage2.use_cls_cond,
                                 use_txt_cond=config.stage2.use_txt_cond,
                                 model_type=model_type,
                                 hparams=config.stage2.hparams,
                                 hparams_dec=config.stage2.hparams_dec)
        elif 'multilevel-hq' in config.stage2.type:
            self.stage2 = HQTransformer(vocab_sizes=config.stage2.vocab_sizes_img,
                                        vocab_size_txt=config.stage2.vocab_size_txt,
                                        decoding_type=config.stage2.decoding_type,
                                        use_cls_cond=config.stage2.use_cls_cond,
                                        use_txt_cond=config.stage2.use_txt_cond,
                                        hparams=config.stage2.hparams,
                                        hparams_dec=config.stage2.hparams_dec)
        else:
            raise ValueError()

        self.config = config
        self.use_cls_cond = config.stage2.use_cls_cond
        self.use_txt_cond = config.stage2.use_txt_cond
        self.type = config.stage2.type
        self.gamma_focal_loss = config.stage2.gamma_focal_loss
        self.temp_soft_labels = config.stage2.temp_soft_labels
        if self.temp_soft_labels is not None:
            self.use_soft_label = True
        else:
            self.use_soft_label = False

        if self.config.stage2.weight_bottom is None:
            self.w_bottom = 1.0
            self.weight_image = 1.0
        else:
            self.w_bottom = self.config.stage2.weight_bottom
            self.w_image = 1.0 + self.w_bottom

        if self.use_txt_cond:
            self.weight_img = config.stage2.weight_img
            self.weight_txt = config.stage2.weight_txt * self.w_image  # compensate the scale of loss_img

        # make the parameters in stage 1 not trainable
        self.stage1.eval()
        for p in self.stage1.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls,
                        path_upstream: str,
                        path_downstream: str) -> Tuple[nn.Module, OmegaConf]:
        config_base = get_base_config(use_default=False)
        config_down = OmegaConf.load(path_downstream)
        config_down = OmegaConf.merge(config_base, config_down)

        model = cls(config_down)
        try:
            model.stage1.from_ckpt(os.path.join(path_upstream, 'stage1_last.ckpt'), strict=False)
        except:
            try:
                model.stage1.from_ckpt(os.path.join(path_upstream, 'ckpt', 'last.ckpt'), strict=False)
            except:
                assert (False)
        if config_down.stage2.use_pretrained:
            if model.type == 'top':
                model.stage2.from_ckpt(os.path.join(path_upstream, 'stage2_last_top.ckpt'),
                                       strict=False,
                                       ignore_keys=['tok_emb_img.weight'])
            elif model.type == 'bottom':
                model.stage2.from_ckpt(os.path.join(path_upstream, 'stage2_last_bottom.ckpt'),
                                       strict=False,
                                       ignore_keys=['tok_emb_img.weight',
                                                    'tok_emb_txt.weight',
                                                    'pos_emb_txt.weight',
                                                    'head_img.weight'])

        return model, config_down

    def sample(self,
               cls_idx: Optional[int] = None,
               top_k: int = 256,
               top_p: Optional[float] = None,
               softmax_temperature: float = 1.0,
               num_candidates: int = 16,
               device: str = 'cuda:0',
               use_fp16: bool = True,
               is_tqdm: bool = True) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        if cls_idx is None:
            sos = self.stage2.sos.repeat(num_candidates, 1, 1)
        else:
            sos = torch.LongTensor([cls_idx]).to(device=device)
            sos = sos.repeat(num_candidates)
            sos = self.stage2.sos(sos).unsqueeze(1)

        codes = sampling_igpt(self.stage2,
                              sos=sos,
                              top_k=top_k,
                              top_p=top_p,
                              softmax_temperature=softmax_temperature,
                              use_fp16=use_fp16,
                              is_tqdm=is_tqdm)
        codes_t = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes_t, None) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels

    def forward(self,
                images: torch.FloatTensor,
                labels: Optional[torch.LongTensor] = None) -> torch.FloatTensor:

        if 'multilevel-hq' in self.type:
            return self.forward_multilevel(images, labels)
        else:
            B, C, H, W = images.shape
            with torch.no_grad():
                with autocast(enabled=False):
                    if self.use_soft_label:
                        codes, softs = self.stage1.get_soft_codes(images, temp=self.temp_soft_labels)
                        (codes_t, codes_b) = codes
                        (soft_t, soft_b) = softs
                    else:
                        codes_t, codes_b = self.stage1.get_codes(images)
                        soft_t, soft_b = None, None
                    codes_t = codes_t.detach()
                    codes_b = codes_b.detach()

            codes_t = codes_t.view(B, -1)
            codes_b = codes_b.view(B, -1)

            if self.type == 'top':
                logits = self.stage2(codes_t, labels)
                codes_gt = codes_t
            elif self.type == 'bottom':
                logits = self.stage2(codes_b, codes_t)[0]
                codes_gt = codes_b

            elif 'hq-transformer' in self.type:
                codes_gt = (codes_t, codes_b)
                logits = self.stage2(codes_gt, labels, self.stage1)

            return logits, codes_gt, (soft_t, soft_b)

    def forward_multilevel(self,
                           images: torch.FloatTensor,
                           labels: Optional[torch.LongTensor] = None) -> torch.FloatTensor:

        B, C, H, W = images.shape
        with torch.no_grad():
            with autocast(enabled=False):
                if self.use_soft_label:
                    codes, softs = self.stage1.get_soft_codes(images, temp=self.temp_soft_labels)
                else:
                    codes = self.stage1.get_codes(images)
                    softs = [None for i in range(0, len(codes))]
                codes = list(map(lambda c: c.detach().view(B, -1), codes))

        logits = self.stage2(codes, labels, self.stage1)

        return logits, codes, softs

    def _compute_loss(self, logits, codes, targets):
        if self.use_soft_label:
            return soft_target_cross_entropy(logits, targets)
        else:
            return F.cross_entropy(logits, codes)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits, codes, softs = self(images, labels=labels if (self.use_cls_cond or self.use_txt_cond) else None)

        if isinstance(logits, tuple):
            loss_top = self._compute_loss(logits[0].view(-1, logits[0].shape[-1]), codes[0].view(-1), softs[0])
            loss_bot = self._compute_loss(logits[1].view(-1, logits[1].shape[-1]), codes[1].view(-1), softs[1])
            loss_img = loss_top + loss_bot * self.w_bottom

            if self.use_txt_cond:
                loss_txt = F.cross_entropy(logits[2].view(-1, logits[2].shape[-1]), labels[:, 1:].reshape(-1))
                loss = loss_img * self.weight_img + loss_txt * self.weight_txt

                self.log("train/loss_txt", loss_txt, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            else:
                loss = loss_img

            self.log("train/loss_top", loss_top, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train/loss_bot", loss_bot, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train/loss_img", loss_img, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        elif isinstance(logits, list):
            for i, (_logit, _code, _soft) in enumerate(zip(logits, codes, softs)):
                _loss = self._compute_loss(_logit.view(-1, _logit.shape[-1]), _code.view(-1), _soft)
                self.log(f"train/loss_level{i}", _loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

                if i == 0:
                    loss_img = _loss
                else:
                    loss_img += (4**i) * _loss

            if self.use_txt_cond:
                logits_txt = logits[-1]
                loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]), labels[:, 1:].reshape(-1))
                loss = loss_img * self.weight_img + loss_txt * self.weight_txt

                self.log("train/loss_txt", loss_txt, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            else:
                loss = loss_img

            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), codes.view(-1))
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, codes, softs = self(images, labels=labels if (self.use_cls_cond or self.use_txt_cond) else None)

        if isinstance(logits, tuple):
            loss_top = self._compute_loss(logits[0].view(-1, logits[0].shape[-1]), codes[0].view(-1), softs[0])
            loss_bot = self._compute_loss(logits[1].view(-1, logits[1].shape[-1]), codes[1].view(-1), softs[1])
            loss_img = loss_top + loss_bot * self.w_bottom

            if self.use_txt_cond:
                loss_txt = F.cross_entropy(logits[2].view(-1, logits[2].shape[-1]), labels[:, 1:].reshape(-1))
                loss = loss_img * self.weight_img + loss_txt * self.weight_txt

                self.log("val/loss_txt", loss_txt, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            else:
                loss = loss_img

            self.log("val/loss_top", loss_top, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val/loss_bot", loss_bot, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val/loss_img", loss_img, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        elif isinstance(logits, list):
            if self.use_txt_cond:
                logits_txt = logits[-1]
                logits = logits[0:len(codes)]

            weights = [4**i for i in range(0, len(logits))]
            _func = lambda _l, _c, _s, _w: _w*self._compute_loss(_l.view(-1, _l.shape[-1]), _c.view(-1), _s)
            losses = list(map(_func, logits, codes, softs, weights))
            loss_img = sum(losses)

            if self.use_txt_cond:
                loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]), labels[:, 1:].reshape(-1))
                loss = loss_img * self.weight_img + loss_txt * self.weight_txt

                self.log("val/loss_txt", loss_txt, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            else:
                loss = loss_img

            for i in range(0, len(codes)):
                loss_level_i = losses[i]/weights[i]
                self.log(f"val/loss_level{i}", loss_level_i, on_step=True, on_epoch=True, prog_bar=False, logger=True)

            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), codes.view(-1))
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def configure_optimizers(self):
        assert self.config.optimizer.opt_type == 'adamW'
        assert self.config.optimizer.sched_type == 'cosine'

        # code segment from minDALL-E
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, SimRQGAN2Generator, HQVAEGenerator)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Parameter)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn == 'sos' or pn == 'sos_depth' or pn == 'cls_token' or pn == 'pos_emb_bot':
                    no_decay.add(fpn)
                elif pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.config.optimizer.weight_decay
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]

        opt = torch.optim.AdamW(optim_groups,
                                lr=self.config.optimizer.base_lr,
                                betas=self.config.optimizer.betas)

        final_steps = self.config.optimizer.max_steps
        if self.config.experiment.epochs == 0:
            steps_per_epoch = 1
        else:
            steps_per_epoch = final_steps // self.config.experiment.epochs
        sched = build_scheduler(opt, self.config.optimizer.base_lr,
                                steps_per_epoch,
                                final_steps,
                                self.config.optimizer.warmup,
                                self.config.optimizer.sched_type)

        sched = {
            'scheduler': sched,
            'name': 'cosine'
        }
        return [opt], [sched]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_epoch_start(self):
        self.stage1.eval()
