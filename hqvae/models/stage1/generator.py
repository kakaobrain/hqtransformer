# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import torch
from typing import Tuple, List, Optional
from omegaconf import OmegaConf
from einops import rearrange

from .modules.layers import Encoder, Decoder, Upsample
from .modules.quantizer import VectorQuantizer, EMAVectorQuantizer
from hqvae.models.stage2.layers import DecoderBlock


class VQGANGenerator(torch.nn.Module):
    def __init__(self,
                 n_embed: int,
                 embed_dim: int,
                 ema_update: bool,
                 hparams: OmegaConf):
        super().__init__()

        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)

        if ema_update:
            self.quantize = EMAVectorQuantizer(dim=embed_dim, n_embed=n_embed, beta=0.25)
        else:
            self.quantize = VectorQuantizer(dim=embed_dim, n_embed=n_embed, beta=0.25)

        self.quant_conv = torch.nn.Conv2d(hparams.z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, hparams.z_channels, 1)

        self.latent_dim = hparams.attn_resolutions[0]

    def forward(self, x, global_step=None):
        quant, diff, code = self.encode(x)
        assert quant.shape[-1] == self.latent_dim, f'latent dim should be [C, {self.latent_dim}, {self.latent_dim}]'
        dec = self.decode(quant)
        return dec, diff, code

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, diff, code = self.quantize(h)
        return quant, diff, (code)

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code: torch.LongTensor, global_step: Optional[int] = None) -> torch.FloatTensor:
        quant = self.quantize.get_codebook_entry(code)
        quant = quant.permute(0, 3, 1, 2)
        dec = self.decode(quant)
        return dec

    def get_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
        h = self.encoder(x)
        h = self.quant_conv(h)
        codes = self.quantize(h)[-1].view(x.shape[0], self.latent_dim ** 2)
        return codes


class VQGAN2Generator(torch.nn.Module):
    decoding_type = {'concat', 'sum'}

    def __init__(self,
                 n_embed: int,
                 embed_dim: int,
                 ema_update: bool,
                 hparams: OmegaConf,
                 hparams_aux: OmegaConf):
        super().__init__()

        assert hparams.z_channels % 2 == 0
        assert hparams_aux.decoding_type in self.decoding_type

        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(ch=hparams.ch,
                               out_ch=hparams.out_ch,
                               ch_mult=hparams.ch_mult[:-1],
                               num_res_blocks=hparams.num_res_blocks,
                               attn_resolutions=[hparams.attn_resolutions[0] * 2],
                               pdrop=hparams.pdrop,
                               in_channels=hparams.in_channels,
                               resolution=hparams.resolution,
                               z_channels=hparams.z_channels,
                               double_z=hparams.double_z,
                               use_init_downsample=hparams.use_init_downsample,
                               use_mid_block=hparams.use_mid_block,
                               use_attn=hparams.use_attn)
        self.decoder_top = Decoder(ch=hparams.ch,
                                   out_ch=hparams.z_channels,
                                   ch_mult=[1, hparams.ch_mult[-1]],
                                   num_res_blocks=hparams.num_res_blocks,
                                   attn_resolutions=hparams.attn_resolutions,
                                   pdrop=hparams.pdrop,
                                   in_channels=hparams.in_channels,
                                   resolution=hparams.attn_resolutions[0] * 2,
                                   z_channels=hparams.z_channels,
                                   double_z=hparams.double_z,
                                   use_init_downsample=False,
                                   use_mid_block=hparams.use_mid_block,
                                   use_attn=hparams.use_attn)

        if ema_update:
            self.quantize_t = EMAVectorQuantizer(dim=embed_dim, n_embed=n_embed, beta=0.25)
            self.quantize_b = EMAVectorQuantizer(dim=embed_dim, n_embed=n_embed, beta=0.25)
        else:
            self.quantize_t = VectorQuantizer(dim=embed_dim, n_embed=n_embed, beta=0.25)
            self.quantize_b = VectorQuantizer(dim=embed_dim, n_embed=n_embed, beta=0.25)

        ch_ratio = 2 if hparams_aux.decoding_type == 'concat' else 1
        self.quant_conv_t = torch.nn.Conv2d(hparams.z_channels, embed_dim, 1)
        self.quant_conv_b = torch.nn.Conv2d(hparams.z_channels * ch_ratio, embed_dim, 1)

        if hparams_aux.upsample == 'deconv2d':
            self.upsample_t = torch.nn.ConvTranspose2d(embed_dim,
                                                       hparams.z_channels // ch_ratio, 4, stride=2, padding=1)
        elif hparams_aux.upsample == 'nearest':
            self.upsample_t = torch.nn.Sequential(
                torch.nn.Conv2d(embed_dim, hparams.z_channels // ch_ratio, 3, 1, 1),
                Upsample(in_channels=None, with_conv=False)
            )
        else:
            raise ValueError(f'{hparams.upsample} is not a supported upsample mode')

        self.post_quant_conv_t = torch.nn.Conv2d(embed_dim, hparams.z_channels, 1)
        self.post_quant_conv_b = torch.nn.Conv2d(embed_dim, hparams.z_channels // ch_ratio, 1)

        self.latent_dim = hparams.attn_resolutions[0]
        self._shared_codebook = hparams_aux.shared_codebook
        self._bottom_start = hparams_aux.bottom_start
        self._decoding_type = hparams_aux.decoding_type

    def forward(self,
                x: torch.FloatTensor,
                global_step: Optional[int] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        quant_t, quant_b, diff_t, diff_b, code = self.encode(x)
        dec = self.decode(quant_t, quant_b, global_step)
        diff = (diff_t, diff_b)
        return dec, diff, code

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h_t, h_b = self.encoder(x, ret_bottom=True)
        h_t = self.quant_conv_t(h_t)
        quant_t, diff_t, code_t = self.quantize_t(h_t)

        d_b = self.decoder_top(self.post_quant_conv_t(quant_t))
        h_b = torch.cat([h_b, d_b], dim=1) if self._decoding_type == 'concat' else h_b + d_b
        h_b = self.quant_conv_b(h_b)
        if self._shared_codebook:
            quant_b, diff_b, code_b = self.quantize_t(h_b)
        else:
            quant_b, diff_b, code_b = self.quantize_b(h_b)
        code = (code_t, code_b)
        return quant_t, quant_b, diff_t, diff_b, code

    def decode(self,
               quant_t: torch.FloatTensor,
               quant_b: torch.FloatTensor,
               global_step: Optional[int] = None) -> torch.FloatTensor:
        quant_t = self.upsample_t(quant_t)
        quant_b = self.post_quant_conv_b(quant_b)
        if global_step is not None and global_step < self._bottom_start and self.training:
            quant_b = torch.zeros_like(quant_b, dtype=quant_b.dtype, device=quant_b.device)
        quant = torch.cat([quant_t, quant_b], dim=1) if self._decoding_type == 'concat' else quant_t + quant_b
        dec = self.decoder(quant)
        return dec


class SimRQGAN2Generator(torch.nn.Module):
    decoding_type = {'concat'}

    def __init__(self,
                 n_embed: int,
                 embed_dim: int,
                 ema_update: bool,
                 hparams: OmegaConf,
                 hparams_aux: OmegaConf):
        super().__init__()

        assert hparams.z_channels % 2 == 0
        assert hparams_aux.decoding_type in self.decoding_type

        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)

        # down / upsampling methods for top code
        if hparams_aux.upsample is None:
            kernel_size = 2
            # baseline
            self.down_t = torch.nn.AvgPool2d(2)
            self.upsample_t = Upsample(in_channels=None, with_conv=False)

            embed_dim_top, embed_dim_bot = embed_dim, embed_dim

        # (average pooling and nearest neighborhood interp.)
        if 'nearest' in hparams_aux.upsample:
            if 'nearest' in hparams_aux.upsample:
                interp_type = 'nearest'
            else:
                assert (False)
            str_kernel_size = hparams_aux.upsample.split(interp_type)[-1]
            if len(str_kernel_size) > 0:
                kernel_size = int(str_kernel_size)
            else:
                kernel_size = 2

            embed_dim_top, embed_dim_bot = embed_dim, embed_dim
            self.down_t = torch.nn.AvgPool2d(kernel_size)
            self.upsample_t = Upsample(in_channels=None,
                                       with_conv=False,
                                       scale=float(kernel_size),
                                       interp_type=interp_type)

        elif 'pixelshuffle' in hparams_aux.upsample:
            str_kernel_size = hparams_aux.upsample.split('pixelshuffle')[-1]
            if len(str_kernel_size) > 0:
                kernel_size = int(str_kernel_size)
            else:
                kernel_size = 2

            self.down_t = torch.nn.PixelUnshuffle(kernel_size)
            self.upsample_t = torch.nn.PixelShuffle(kernel_size)

            embed_dim_top, embed_dim_bot = embed_dim * kernel_size * kernel_size, embed_dim

        elif 'conv' in hparams_aux.upsample:
            kernel_size = int(hparams_aux.upsample.split('conv')[-1])
            self.down_t = torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size, stride=kernel_size, padding=0)
            self.upsample_t = torch.nn.ConvTranspose2d(embed_dim,
                                                       embed_dim,
                                                       kernel_size=kernel_size,
                                                       stride=kernel_size,
                                                       padding=0)

            embed_dim_top, embed_dim_bot = embed_dim, embed_dim

        self.bottom_window = kernel_size

        if ema_update:
            self.quantize_t = EMAVectorQuantizer(dim=embed_dim_top, n_embed=n_embed, beta=0.25)
            self.quantize_b = EMAVectorQuantizer(dim=embed_dim_bot, n_embed=n_embed, beta=0.25)
        else:
            self.quantize_t = VectorQuantizer(dim=embed_dim_top, n_embed=n_embed, beta=0.25)
            self.quantize_b = VectorQuantizer(dim=embed_dim_bot, n_embed=n_embed, beta=0.25)

        self.quant_conv_b = torch.nn.Conv2d(hparams.z_channels, embed_dim, 1)
        self.post_quant_conv_b = torch.nn.Conv2d(embed_dim*2, hparams.z_channels, 1)

        self.latent_dim = hparams.attn_resolutions[0]
        self._decoding_type = hparams_aux.decoding_type
        self._bottom_start = hparams_aux.bottom_start
        self._shared_codebook = hparams_aux.shared_codebook

    def forward(self,
                x: torch.FloatTensor,
                global_step: Optional[int] = None
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        if (global_step is not None and
           global_step >= self._bottom_start and
           self.training):

            quant_t, quant_b, diff_t, diff_b, code = self.encode(x)
            dec_tb = self.decode(quant_t, quant_b, global_step)

            empty_quant_b = torch.zeros_like(quant_b, dtype=quant_b.dtype, device=quant_b.device)
            dec_t = self.decode(quant_t, empty_quant_b, global_step)
            dec = (dec_t, dec_tb)

        else:
            quant_t, quant_b, diff_t, diff_b, code = self.encode(x)
            dec = self.decode(quant_t, quant_b, global_step)

        diff = (diff_t, diff_b, torch.abs(code[2]).mean())

        return dec, diff, code

    # visualization purpose
    def forward_topbottom(self,
                          x: torch.FloatTensor,
                          global_step: Optional[int] = None
                          ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        quant_t, quant_b, diff_t, diff_b, code = self.encode(x)
        dec_t = self.decode(quant_t, torch.zeros_like(quant_b, dtype=quant_b.dtype, device=quant_b.device), global_step)
        dec_b = self.decode(torch.zeros_like(quant_t, dtype=quant_t.dtype, device=quant_t.device), quant_b, global_step)
        dec_tb = self.decode(quant_t, quant_b, global_step)
        diff = (diff_t, diff_b)
        return (dec_t, dec_b, dec_tb), diff, code

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h_b = self.quant_conv_b(self.encoder(x))
        h_t = self.down_t(h_b)
        quant_t, diff_t, code_t = self.quantize_t(h_t)
        h_b = h_b - self.upsample_t(quant_t)

        if self._shared_codebook:
            quant_b, diff_b, code_b = self.quantize_t(h_b)
        else:
            quant_b, diff_b, code_b = self.quantize_b(h_b)
        code = (code_t, code_b, h_b)

        return quant_t, quant_b, diff_t, diff_b, code

    def decode(self,
               quant_t: torch.FloatTensor,
               quant_b: torch.FloatTensor,
               global_step: Optional[int] = None) -> torch.FloatTensor:
        quant_t = self.upsample_t(quant_t)

        quant = torch.cat([quant_t, quant_b], dim=1)
        quant = self.post_quant_conv_b(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self,
                    code_t: torch.LongTensor,
                    code_b: torch.LongTensor) -> torch.FloatTensor:
        assert code_t is not None or code_b is not None

        if code_t is None and code_b is not None:
            quant_b = self.quantize_b.get_codebook_entry(code_b)
            quant_b = quant_b.permute(0, 3, 1, 2)

            if isinstance(self.down_t, torch.nn.AvgPool2d):
                num_chn = quant_b.shape[1]
            elif isinstance(self.down_t, torch.nn.Conv2d):
                num_chn = quant_b.shape[1]
            elif isinstance(self.down_t, torch.nn.PixelUnshuffle):
                num_chn = quant_b.shape[1]*(self.bottom_window * self.bottom_window)

            quant_t = torch.zeros(quant_b.shape[0], num_chn,
                                  quant_b.shape[2]//self.bottom_window, quant_b.shape[3]//self.bottom_window,
                                  dtype=quant_b.dtype,
                                  device=quant_b.device)

        elif code_t is not None and code_b is None:
            quant_t = self.quantize_t.get_codebook_entry(code_t)
            quant_t = quant_t.permute(0, 3, 1, 2)

            if isinstance(self.down_t, torch.nn.AvgPool2d):
                num_chn = quant_t.shape[1]
            elif isinstance(self.down_t, torch.nn.Conv2d):
                num_chn = quant_t.shape[1]
            elif isinstance(self.down_t, torch.nn.PixelUnshuffle):
                num_chn = quant_t.shape[1]//(self.bottom_window * self.bottom_window)

            quant_b = torch.zeros(quant_t.shape[0], num_chn,
                                  quant_t.shape[2]*self.bottom_window, quant_t.shape[3]*self.bottom_window,
                                  dtype=quant_t.dtype,
                                  device=quant_t.device)

        else:
            quant_t = self.quantize_t.get_codebook_entry(code_t)
            quant_t = quant_t.permute(0, 3, 1, 2)
            quant_b = self.quantize_b.get_codebook_entry(code_b)
            quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)
        return dec

    def get_codes(self, x):
        return (self.encode(x)[-1][0], self.encode(x)[-1][1])

    @torch.no_grad()
    def get_soft_codes(self, xs, temp=1.0, stochastic=False):
        h_b = self.quant_conv_b(self.encoder(xs))
        h_t = self.down_t(h_b)
        quant_t, diff_t, code_t, soft_t = self.quantize_t.get_soft_codes(h_t, temp, stochastic)

        h_b = h_b - self.upsample_t(quant_t)

        if self._shared_codebook:
            quant_b, diff_b, code_b, soft_b = self.quantize_t.get_soft_codes(h_b, temp, stochastic)
        else:
            quant_b, diff_b, code_b, soft_b = self.quantize_b.get_soft_codes(h_b, temp, stochastic)
        code = (code_t, code_b)
        soft_code = (soft_t, soft_b)

        return code, soft_code

    def from_ckpt(self, path: str, strict: bool = True) -> None:
        ckpt_ = torch.load(path, map_location='cpu')['state_dict']
        ckpt = {}
        for k, v in ckpt_.items():  # matching keys for backward compatibility
            ckpt[k[10:]] = v
        self.load_state_dict(ckpt, strict=strict)
        print(f'{path} successfully restored..')


class HQVAEGenerator(torch.nn.Module):
    decoding_type = {'add', 'concat'}

    def resampling_layers(self, resample_type, embed_dim, cur_level, code_levels):
        # down / upsampling methods for top code
        if resample_type is None:
            kernel_size = 2
            # baseline
            down_t = torch.nn.AvgPool2d(2)
            upsample_t = Upsample(in_channels=None, with_conv=False)

            embed_dim_top, embed_dim_bot = embed_dim, embed_dim

        elif 'nearest' in resample_type:
            if 'nearest' in resample_type:
                interp_type = 'nearest'
            else:
                assert (False)
            str_kernel_size = resample_type.split(interp_type)[-1]
            if len(str_kernel_size) > 0:
                kernel_size = int(str_kernel_size)
            else:
                kernel_size = 2

            embed_dim_top, embed_dim_bot = embed_dim, embed_dim
            down_t = torch.nn.AvgPool2d(kernel_size)
            upsample_t = Upsample(in_channels=None, with_conv=False, scale=float(kernel_size), interp_type=interp_type)

        elif 'pixelshuffle' in resample_type:
            str_kernel_size = resample_type.split('pixelshuffle')[-1]
            if len(str_kernel_size) > 0:
                kernel_size = int(str_kernel_size)
            else:
                kernel_size = 2

            down_t = torch.nn.PixelUnshuffle(kernel_size)
            upsample_t = torch.nn.PixelShuffle(kernel_size)

            embed_dim_top = embed_dim * (kernel_size * kernel_size) ** (code_levels - cur_level - 1)
            embed_dim_bot = embed_dim

        elif 'conv' in resample_type:
            kernel_size = int(resample_type.split('conv')[-1])
            down_t = torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size, stride=kernel_size, padding=0)
            upsample_t = torch.nn.ConvTranspose2d(embed_dim, embed_dim,
                                                  kernel_size=kernel_size,
                                                  stride=kernel_size,
                                                  padding=0)

            embed_dim_top, embed_dim_bot = embed_dim, embed_dim

        return down_t, upsample_t, embed_dim_top, embed_dim_bot, kernel_size

    def __init__(self,
                 n_embed_levels: int,
                 embed_dim: int,
                 ema_update: bool,
                 hparams: OmegaConf,
                 hparams_aux: OmegaConf):
        super().__init__()

        assert hparams.z_channels % 2 == 0
        assert hparams_aux.decoding_type in self.decoding_type

        if (hparams_aux.restart_unused_codes is not None):
            restart_unused_codes = hparams_aux.restart_unused_codes
        else:
            restart_unused_codes = False

        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)

        self.code_levels = hparams_aux.code_levels

        self.bottom_window = 1
        self.downsamples = []
        self.upsamples = []
        self.quantizers = []

        for ci in range(0, self.code_levels-1):
            down_t, upsample_t, embed_dim_top, embed_dim_bot, kernel_size = self.resampling_layers(hparams_aux.upsample,
                                                                                                   embed_dim, ci,
                                                                                                   self.code_levels)
            self.downsamples.append(down_t)
            self.upsamples.append(upsample_t)

            self.bottom_window *= kernel_size

            if ema_update:
                self.quantizers.append(EMAVectorQuantizer(dim=embed_dim_top,
                                                          n_embed=n_embed_levels[ci],
                                                          beta=0.25,
                                                          restart_unused_codes=restart_unused_codes))
            else:
                self.quantizers.append(VectorQuantizer(dim=embed_dim_top, n_embed=n_embed_levels[ci], beta=0.25))

        if ema_update:
            self.quantizers.append(EMAVectorQuantizer(dim=embed_dim_bot,
                                                      n_embed=n_embed_levels[-1],
                                                      beta=0.25,
                                                      restart_unused_codes=restart_unused_codes))
        else:
            self.quantizers.append(VectorQuantizer(dim=embed_dim_bot, n_embed=n_embed_levels[-1], beta=0.25))

        self.downsamples = torch.nn.ModuleList(self.downsamples)
        self.upsamples = torch.nn.ModuleList(self.upsamples)
        self.quantizers = torch.nn.ModuleList(self.quantizers)
        self.quant_conv_b = torch.nn.Conv2d(hparams.z_channels, embed_dim, 1)
        self.post_quant_conv_b = torch.nn.Conv2d(embed_dim, hparams.z_channels, 1)

        self.latent_dim = hparams.attn_resolutions[0]
        self._decoding_type = hparams_aux.decoding_type

        self._shared_codebook = hparams_aux.shared_codebook

        self._bottom_start = hparams_aux.bottom_start

        self.embed_dim = embed_dim

    def forward(self,
                x: torch.FloatTensor,
                global_step: Optional[int] = None
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        quant, diffs, codes, resids = self.encode(x)
        dec = self.decode(quant, global_step)

        # compute residual loss
        resid_loss = []
        for _resid in resids:
            resid_loss.append(_resid.mean())

        return dec, diffs, codes + [sum(resid_loss)]

    def encode(self,
               x: torch.FloatTensor,
               soft_codes: bool = False,
               temp: int = 1.0,
               stochastic: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        h_map = []
        h_map.append(self.quant_conv_b(self.encoder(x)))

        for downsample in self.downsamples:
            h_map.insert(0, downsample(h_map[0]))

        resids = []
        diffs = []
        codes = []
        recons = [0]
        softs = []
        for qi, quantizer in enumerate(self.quantizers):
            _resid = h_map[qi] - recons[-1]
            if (soft_codes):
                _quant, _diff, _code, _scode = quantizer.get_soft_codes(_resid, temp, stochastic)
                softs.append(_scode)
            else:
                _quant, _diff, _code = quantizer(_resid)
            _recon = _quant + recons[-1]
            if (qi < len(self.upsamples)):
                _recon = self.upsamples[qi](_recon)

            resids.append(_resid)
            recons.append(_recon)
            diffs.append(_diff)
            codes.append(_code)

        if (soft_codes):
            return recons[-1], diffs, softs, codes, resids[1:]
        else:
            return recons[-1], diffs, codes, resids[1:]

    def decode(self,
               quant: torch.FloatTensor,
               global_step: Optional[int] = None) -> torch.FloatTensor:
        quant = self.post_quant_conv_b(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self,
                    codes: List[torch.LongTensor]) -> torch.FloatTensor:
        B = 1
        for code in codes:
            if code is not None:
                B = code.size(0)
                device = code.device

        quant = 0
        for hi, (code, quantizer) in enumerate(zip(codes, self.quantizers)):
            if code is not None:
                _quant = quantizer.get_codebook_entry(code)
                _quant = _quant.permute(0, 3, 1, 2)
            else:
                K = int(self.latent_dim // (2 ** (self.code_levels-hi-1)))
                _quant = torch.zeros(B, quantizer.dim, K, K, device=device)

            quant = quant + _quant
            if (hi < len(self.upsamples)):
                quant = self.upsamples[hi](quant)

        dec = self.decode(quant)
        return dec

    def get_codes(self, x):
        return self.encode(x)[2]

    @torch.no_grad()
    def get_soft_codes(self, xs, temp=1.0, stochastic=False):
        recons, diffs, softs, codes, resids = self.encode(xs, True, temp, stochastic)
        return codes, softs

    def from_ckpt(self, path: str, strict: bool = True) -> None:
        ckpt_ = torch.load(path, map_location='cpu')['state_dict']
        ckpt = {}
        for k, v in ckpt_.items():  # matching keys for backward compatibility
            ckpt[k[10:]] = v
        self.load_state_dict(ckpt, strict=strict)
        print(f'{path} successfully restored..')
