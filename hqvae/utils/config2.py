# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from typing import Optional, List
from dataclasses import dataclass, field
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    dataset: Optional[str] = None
    tokenizer_type: Optional[str] = 'bpe16k_huggingface'
    context_length: Optional[int] = 64
    image_resolution: int = 256
    transforms: str = 'dalle-vqvae'
    bpe_pdrop: Optional[float] = 0.1


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
class VQGAN2Hparams:
    upsample: Optional[str] = None
    shared_codebook: Optional[bool] = None
    bottom_start: Optional[int] = 100000000000 # no bypass
    decoding_type: str = 'concat'
    restart_unused_codes: Optional[bool] = None
    code_levels: Optional[int] = None


@dataclass
class Stage2Hparams:
    embed_dim: int = 1536
    n_layers: int = 42
    n_heads: int = 24
    n_dense_layers: int = 42
    ctx_len: Optional[int] = None
    ctx_len_img: Optional[int] = 256
    ctx_len_txt: Optional[int] = 64
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    mlp_bias: bool = True
    attn_bias: bool = True
    gelu_use_approx: bool = False
    use_head_txt: bool = True
    n_classes: Optional[int] = None
    causal_attn: Optional[str] = None
    embedding_type: Optional[str] = 'baseline'
    position_embedding: Optional[str] = '1d'
    bottom_head_type: Optional[str] = 'linear'
    use_random_order: Optional[bool] = False
    rate_random_order: Optional[float] = 1.0


@dataclass
class Stage1Config:
    type: str = 'vqgan'
    embed_dim: int = 256
    n_embed: int = 16384
    n_embed_levels: List[int] = field(default_factory=lambda: [8192, 8192, 8192])
    ema_update: bool = False
    hparams: Stage1Hparams = Stage1Hparams()
    hparams_aux: Optional[VQGAN2Hparams] = None


@dataclass
class Stage2Config:
    type: str = 'transformer1d'
    vocab_size_txt: int = 16384
    vocab_size_img: int = 16384
    vocab_sizes_img: List[int] = field(default_factory=lambda: [8192, 8192, 8192])
    decoding_type: Optional[str] = None
    ratio_bot2top: Optional[int] = 4
    use_pretrained: Optional[bool] = False
    use_cls_cond: Optional[bool] = None
    use_txt_cond: Optional[bool] = None
    use_pretrained: Optional[bool] = False
    weight_bottom: Optional[float] = 4.0  # bot code length is 4 * top code length
    weight_txt: Optional[float] = None
    weight_img: Optional[float] = None
    gamma_focal_loss: Optional[float] = None
    temp_soft_labels: Optional[float] = None
    use_l2norm_logits: Optional[bool] = None
    hparams: Optional[Stage2Hparams] = None  # Stage2Hparams()
    hparams_enc: Optional[Stage2Hparams] = None  # Stage2Hparams()
    hparams_dec: Optional[Stage2Hparams] = None  # Stage2Hparams()


@dataclass
class WarmupConfig:
    warmup_epoch: int = 1
    multiplier: int = 1
    buffer_epoch: int = 0
    min_lr: float = 0.0
    mode: str = 'fix'
    peak_lr: float = 1e-4
    start_from_zero: bool = True


@dataclass
class OptConfig:
    opt_type: str = 'adamW'
    base_lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.99])
    grad_clip_norm: float = 1.0

    sched_type: str = 'cosine'
    max_steps: int = 0
    min_lr: float = 0.0
    init_lr: float = 0.0

    warmup: Optional[WarmupConfig] = None


@dataclass
class ExpConfig:
    local_batch_size: int = 4
    total_batch_size: int = 512
    valid_batch_size: int = 32
    epochs: int = 0
    save_ckpt_freq: int = 1
    test_freq: int = 1
    use_amp: bool = True


@dataclass
class DefaultConfig:
    dataset: DataConfig = DataConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()


@dataclass
class FineTuningConfig:
    dataset: DataConfig = DataConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    optimizer: OptConfig = OptConfig()
    experiment: ExpConfig = ExpConfig()


def get_base_config(use_default=True):
    return OmegaConf.structured(DefaultConfig if use_default else FineTuningConfig)
