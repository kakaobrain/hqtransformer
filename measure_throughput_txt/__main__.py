# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from RQ-Transformer
# https://github.com/kakaobrain/rq-vae-transformer/blob/main/measure_throughput/__main__.py
# ------------------------------------------------------------------------------------

import os
import torch
import platform
import time
import torchvision.transforms as transforms

from einops import rearrange
from dataclasses import dataclass
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from hqvae.datasets import CC3MTextOnly
from hqvae.models import ImageGPT2
from hqvae.utils.config2 import get_base_config
from hqvae.utils.sampling import sampling_ihqgpt


def get_text_loader(args, config):
    valid_transform = [
        transforms.Resize(size=(config.dataset.image_resolution, config.dataset.image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    if args.dataset == 'cc3m':
        dataset_val = CC3MTextOnly(
            split='val',
            tok_name=config.dataset.tokenizer_type,
            transform=valid_transform,
            context_length=config.dataset.context_length,
            dropout=None,
        )
    else:
        raise NotImplementedError
    loader = DataLoader(dataset_val, shuffle=False, pin_memory=True,
                        batch_size=args.batch_size, num_workers=16)
    return loader


def load_model(result_path, device='cuda'):
    if 'ckpt' in result_path:
        config_path = os.path.join(os.path.dirname(result_path), '..', 'config.yaml')
    else:
        config_path = os.path.join(result_path, 'config.yaml')
        result_path = os.path.join(result_path, 'ckpt/last.ckpt')
    print(result_path)

    config_base = get_base_config(use_default=False)
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(config_base, config)
    model = ImageGPT2(config)

    return model, config


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
    bot_resolution: int = 16
    dataset: str = 'cc3m'


def main(args: Experiment):
    torch.set_grad_enabled(False)

    model_ar, config = load_model(args.model_path)

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

    kerH = args.bot_resolution // args.top_resolution

    def loop(loop_idx: int):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter_per_loop)]
        middles = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter_per_loop)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter_per_loop)]

        torch.cuda.synchronize(device)
        tic = time.time()

        loader = get_text_loader(args, config)
        pbar = tqdm(enumerate(loader), total=n_iter_per_loop)
        for i, (_, txts) in pbar:
            if (i >= n_iter_per_loop):
                break
            txts = txts.cuda()
            starts[i].record()
            codes_t, codes_b = sampling_ihqgpt(model_ar.stage2,
                                               cond=txts,
                                               num_candidates=1,
                                               top_k_top=2048,
                                               top_p_top=1.0,
                                               top_k_bot=2048,
                                               top_p_bot=1.0,
                                               softmax_temperature=1.0,
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
