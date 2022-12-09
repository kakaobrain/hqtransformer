# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
from datetime import datetime
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    dataset: Optional[str] = None
    image_resolution: int = 256


@dataclass
class Stage1Hparams:
    double_z: bool = False
    z_channels: int = 256
    resolution: int = 256
    in_channels: int = 3
    out_ch: int = 3
    ch: int = 128
    ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    num_res_blocks: int = 2
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    pdrop: float = 0.0
    use_init_downsample: bool = False
    use_mid_block: bool = True
    use_attn: bool = True


@dataclass
class Stage1HparamsDisc:
    disc_conditional: bool = False
    disc_in_channels: int = 3
    disc_start: int = 0
    disc_weight: float = 0.75
    disc_num_layers: int = 2
    codebook_weight: float = 1.0
    norm_type: str = 'bn'  # [bn, actnorm, gn] BatchNorm2D, ActNorm, GroupNorm(#groups=16)
    residual_l1_weight: Optional[float] = None
    use_recon_top: Optional[bool] = True
    use_perceptual_top: Optional[bool] = False
    use_adversarial_top: Optional[bool] = False


@dataclass
class VQGAN2Hparams:
    upsample: Optional[str] = None
    shared_codebook: Optional[bool] = None
    bottom_start: Optional[int] = None
    decoding_type: str = 'concat'

    restart_unused_codes: Optional[bool] = None
    # multi-level HQ
    code_levels: Optional[int] = None

    # SIVAE
    code_num_k: Optional[int] = None
    n_layers: int = 4
    n_heads: int = 16
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    mlp_bias: bool = True
    attn_bias: bool = True
    gelu_use_approx: bool = False


@dataclass
class Stage1Config:
    type: str = 'vqgan'
    embed_dim: int = 256
    n_embed: int = 16384
    n_embed_levels: List[int] = field(default_factory=lambda: [8192, 8192, 8192])
    ema_update: bool = False
    hparams: Stage1Hparams = Stage1Hparams()
    hparams_disc: Optional[Stage1HparamsDisc] = Stage1HparamsDisc()
    hparams_aux: Optional[VQGAN2Hparams] = None


@dataclass
class WarmupConfig:
    multiplier: float = 1.0
    warmup_epoch: float = 0.0
    buffer_epoch: float = 0.0
    min_lr: float = 0.0
    mode: str = 'fix'
    start_from_zero: bool = True


@dataclass
class OptConfig:
    opt_type: str = 'adam'
    betas: Optional[Tuple[float]] = None
    base_lr: float = 1e-4
    use_amp: bool = True
    grad_clip_norm: Optional[float] = 1.0
    max_steps: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    warmup_config: WarmupConfig = WarmupConfig()


@dataclass
class ExpConfig:
    local_batch_size: int = 16
    total_batch_size: int = 512
    valid_batch_size: int = 32
    epochs: int = 100
    save_ckpt_freq: int = 2
    test_freq: int = 1
    img_logging_freq: int = 5000
    fp16_grad_comp: bool = False


@dataclass
class DefaultConfig:
    dataset: DataConfig = DataConfig()
    stage1: Stage1Config = Stage1Config()
    optimizer: OptConfig = OptConfig()
    experiment: ExpConfig = ExpConfig()


def update_config(cfg_base, cfg_new):
    if cfg_new.stage1.type == 'vqgan':
        pass
    elif cfg_new.stage1.type == 'vqgan2':
        cfg_base.stage1.hparams_aux = VQGAN2Hparams()
    elif cfg_new.stage1.type == 'simrqgan2':
        cfg_base.stage1.hparams_aux = VQGAN2Hparams()
    elif cfg_new.stage1.type == 'hqvae':
        cfg_base.stage1.hparams_aux = VQGAN2Hparams()
    elif cfg_new.stage1.type == 'sivae':
        cfg_base.stage1.hparams_aux = VQGAN2Hparams()
    else:
        raise ValueError(f'{cfg_new.stage1.type} not supported..')
    cfg_update = OmegaConf.merge(cfg_base, cfg_new)
    return cfg_update


def build_config(args):
    cfg_base = OmegaConf.structured(DefaultConfig)
    if args.eval:
        cfg_new = OmegaConf.load(os.path.join(args.result_path, 'config.yaml'))
        cfg_update = update_config(cfg_base, cfg_new)
        result_path = args.result_path
    else:
        cfg_new = OmegaConf.load(args.config_path)
        cfg_update = update_config(cfg_base, cfg_new)
        now = datetime.now().strftime('%d%m%Y_%H%M%S')
        result_path = os.path.join(args.result_path,
                                   os.path.basename(args.config_path).split('.')[0],
                                   now)
    return cfg_update, result_path
