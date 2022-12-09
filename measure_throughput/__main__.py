# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from RQ-Transformer
# https://github.com/kakaobrain/rq-vae-transformer/blob/main/measure_throughput/__main__.py
# ------------------------------------------------------------------------------------

import torch
import platform
import time
import random

from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange
from dataclasses import dataclass

from hqvae.models import ImageGPT2
from hqvae.utils.config2 import get_base_config
from hqvae.utils.sampling import sampling_ihqgpt, sampling_hqtransformer


def load_model(result_path, device='cuda'):
    config_base = get_base_config(use_default=False)
    config = OmegaConf.load(result_path)
    config = OmegaConf.merge(config_base, config)
    model = ImageGPT2(config)

    return model


@dataclass
class Experiment:
    f: int = 32
    model: str = 'huge'
    d: int = 4
    c: int = 16384

    batch_size: int = 50

    n_loop: int = 6
    warmup: int = 1

    model_path: str = ''
    top_resolution: int = 8
    code_levels: int = 2


def main(args: Experiment):
    torch.set_grad_enabled(False)

    model_ar = load_model(args.model_path)

    device = torch.device('cuda')
    model_ar = model_ar.to(device)
    model_ar.eval()

    title = (f'bs{args.batch_size}, '
             f'sampling loops {args.warmup+1}-{args.n_loop}'
             )
    print(title)
    print('python: %s, torch: %s, cudnn: %s, cuda: %s, gpu: %s' % (
        platform.python_version(),
        torch.__version__,
        torch.backends.cudnn.version(),
        torch.version.cuda,
        torch.cuda.get_device_name(device)
    ))

    ar_size = sum([p.numel() for p in model_ar.stage2.parameters()]) / (10 ** 6)
    print(f'transformer size: {ar_size:.1f}M')

    batch_size = args.batch_size
    n_iter_per_loop = (1000 + batch_size - 1) // batch_size
    n_loop = args.n_loop

    kerH = 2

    def loop(loop_idx: int):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter_per_loop)]
        middles = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter_per_loop)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter_per_loop)]

        torch.cuda.synchronize(device)
        tic = time.time()

        pbar = tqdm(range(n_iter_per_loop), total=n_iter_per_loop)
        for i in pbar:
            starts[i].record()
            if args.code_levels == 2:
                codes_t, codes_b = sampling_ihqgpt(model_ar.stage2,
                                                   cond=random.randint(0, 999),
                                                   num_candidates=batch_size,
                                                   top_k_top=None,
                                                   top_p_top=None,
                                                   top_k_bot=None,
                                                   top_p_bot=None,
                                                   softmax_temperature=[1.0 for i in range(0, args.code_levels)],
                                                   use_fp16=True,
                                                   is_tqdm=False,
                                                   max_seq_len=args.top_resolution * args.top_resolution,
                                                   model_stage1=None)
                middles[i].record()
                codes_t = rearrange(codes_t, 'B (H W) -> B H W', H=args.top_resolution)  # [B, 16, 16]
                codes_b = rearrange(codes_b, 'B (H W) (kerH kerW) -> B (H kerH) (W kerW)', H=args.top_resolution, kerH=kerH)
                chunks_t = codes_t.chunk(batch_size)
                chunks_b = codes_b.chunk(batch_size)
                pixels = torch.cat([model_ar.stage1.decode_code(chunk_t, chunk_b)
                                    for chunk_t, chunk_b in zip(chunks_t, chunks_b)], dim=0)

                _ = (0.5 * pixels + 0.5).clamp(0, 1)
                ends[i].record()

            elif args.code_levels == 3:
                codes_levels = sampling_hqtransformer(model_ar.stage2,
                                                      num_candidates=batch_size,
                                                      cond=random.randint(0, 999),
                                                      top_k=[None for i in range(0, args.code_levels)],
                                                      top_p=[None for i in range(0, args.code_levels)],
                                                      softmax_temperature=[1.0 for i in range(0, args.code_levels)],
                                                      use_fp16=True,
                                                      is_tqdm=False,
                                                      max_seq_len=args.top_resolution * args.top_resolution,
                                                      model_stage1=None)
                middles[i].record()                                                    
                K = args.top_resolution
                codes_t = rearrange(codes_levels[0], 'B (H W) -> B H W', H=K)  # [B, 16, 16]
                codes_m = rearrange(codes_levels[1], 'B (H W) (kerH kerW) -> B (H kerH) (W kerW)', H=K, W=K, kerH=2)
                codes_b = rearrange(codes_levels[2], 'B (H W) (kerH kerW) -> B (H kerH) (W kerW)', H=K, W=K, kerH=4)
                
                pixels = [model_ar.stage1.decode_code([codes_t[i:i+1],
                                                        codes_m[i:i+1], codes_b[i:i+1]]) for i in range(0, batch_size)]
                pixels = torch.cat(pixels, dim=0) * 0.5 + 0.5
                pixels = torch.clamp(pixels, 0, 1)

                ends[i].record()

            speed = (time.time() - tic) / ((i + 1) * batch_size) * 1000
            pbar.set_description(f'{loop_idx+1}/{n_loop} | {speed:.3f} ms/sample (estimated)')

        torch.cuda.synchronize(device)
        toc = time.time()

        elapsed_time = toc - tic
        elapsed_time_ar = sum([starts[i].elapsed_time(middles[i]) for i in range(n_iter_per_loop)]) / 1000
        elapsed_time_decode = sum([middles[i].elapsed_time(ends[i]) for i in range(n_iter_per_loop)]) / 1000
        print(f'{loop_idx + 1}/{n_loop} | '
              f'{elapsed_time:.1f} s/loop (ar: {elapsed_time_ar:.1f}, decode: {elapsed_time_decode:.1f})')

        speed = elapsed_time / (n_iter_per_loop * batch_size) * 1000
        speed_ar = elapsed_time_ar / (n_iter_per_loop * batch_size) * 1000
        speed_decode = elapsed_time_decode / (n_iter_per_loop * batch_size) * 1000
        print(f'{loop_idx+1}/{n_loop} | {speed:.1f} ms/sample (ar: {speed_ar:.1f}, decode: {speed_decode:.1f})')

        return speed, speed_ar, speed_decode

    speeds = []
    speeds_ar = []
    speeds_decode = []

    print('-' * 80)
    for loop_idx in range(args.n_loop):
        speed, speed_ar, speed_decode = loop(loop_idx)

        if loop_idx < args.warmup:
            continue

        speeds.append(speed)
        speeds_ar.append(speed_ar)
        speeds_decode.append(speed_decode)
    print('-' * 80)

    n = len(speeds)
    speed = sum(speeds) / n
    speed_ar = sum(speeds_ar) / n
    speed_decode = sum(speeds_decode) / n
    print(f'{title} | {speed:.4f} ms/sample (ar: {speed_ar:.4f}, decode: {speed_decode:.4f})')
    print('=' * 80)


if __name__ == '__main__':

    args = OmegaConf.merge(OmegaConf.structured(Experiment()), OmegaConf.from_cli())
    main(args)
