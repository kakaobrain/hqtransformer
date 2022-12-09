# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import sys
import argparse
import pickle
import torch
import numpy as np

from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf

from hqvae.models import ImageGPT2
from hqvae.utils.config2 import get_base_config
from hqvae.utils.utils import set_seed
from hqvae.utils.sampling import sampling_ihqgpt, sampling_hqtransformer


parser = argparse.ArgumentParser()

parser.add_argument('-r', '--result-path', type=str, required=True)
parser.add_argument('-m', '--model-path', type=str, default='', required=True)

parser.add_argument('--top-k', type=int, default=2048, required=False)
parser.add_argument('--top-p', type=float, default=1.0, required=False)
parser.add_argument('--temperature', type=float, default=1.0, required=False)
parser.add_argument('--temperature-decay', type=float, default=1.0, required=False)

parser.add_argument('--batch-size', type=int, default=50, required=False)
parser.add_argument('--code-level', type=int, default=2, required=False)
parser.add_argument('--top-resolution', type=int, default=8, required=False)
parser.add_argument('--bot-resolution', type=int, default=16, required=False)
parser.add_argument('--seed', type=int, default=0, required=False)
parser.add_argument('--num-classes', type=int, default=1000, required=False)


args = parser.parse_args()


def load_model_legacy(result_path, device='cuda'):
    config_base = get_base_config(use_default=False)
    config = OmegaConf.load(os.path.join(result_path, 'config.yaml'))
    config = OmegaConf.merge(config_base, config)

    model = ImageGPT2(config)
    ckpt_ = torch.load(os.path.join(result_path, 'ckpt/last.ckpt'), map_location='cpu')['state_dict']
    ckpt = {}
    for k, v in ckpt_.items():
        if 'stage1' in k:
            ckpt['stage1.'+k[17:]] = v
        else:
            ckpt[k] = v
    model.load_state_dict(ckpt, strict=True)
    model.to(device=device)

    return model


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

    ckpt = torch.load(result_path, map_location='cpu')['state_dict']

    model.load_state_dict(ckpt, strict=True)
    model.to(device=device)

    return model


def save_pickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


def sampling_top(model,
                 cls_idx,
                 top_k,
                 top_p,
                 softmax_temperature,
                 num_candidates,
                 device='cuda',
                 use_fp16=True,
                 is_tqdm=False,
                 top_resolution=8,
                 bot_resolution=16,
                 hilbert_order=None,
                 model_stage1=None):

    kerH = bot_resolution // top_resolution

    codes_t, codes_b = sampling_ihqgpt(model,
                                       cond=cls_idx,
                                       num_candidates=num_candidates,
                                       top_k_top=top_k,
                                       top_p_top=top_p,
                                       top_k_bot=top_k,
                                       top_p_bot=top_p,
                                       softmax_temperature=softmax_temperature,
                                       use_fp16=use_fp16,
                                       is_tqdm=is_tqdm,
                                       max_seq_len=top_resolution * top_resolution,
                                       model_stage1=model_stage1)

    codes_t = rearrange(codes_t, 'B (H W) -> B H W', H=top_resolution)  # [B, 16, 16]
    codes_b = rearrange(codes_b, 'B (H W) (kerH kerW) -> B (H kerH) (W kerW)', H=top_resolution, kerH=kerH)
    return codes_t, codes_b


def sampling_level3(model,
                    cls_idx,
                    top_k,
                    top_p,
                    softmax_temperatures,
                    num_candidates,
                    device='cuda',
                    use_fp16=True,
                    is_tqdm=False,
                    resolutions=16,
                    hilbert_order=None,
                    model_stage1=None):

    codes_levels = sampling_hqtransformer(model,
                                          num_candidates=num_candidates,
                                          cond=cls_idx,
                                          top_k=[top_k for i in range(0, len(resolutions))],
                                          top_p=[top_p for i in range(0, len(resolutions))],
                                          softmax_temperature=softmax_temperatures,
                                          use_fp16=use_fp16,
                                          is_tqdm=is_tqdm,
                                          max_seq_len=resolutions[0] * resolutions[0],
                                          model_stage1=model_stage1)
    K = resolutions[0]
    codes_t = rearrange(codes_levels[0], 'B (H W) -> B H W', H=K)  # [B, 16, 16]
    codes_m = rearrange(codes_levels[1], 'B (H W) (kerH kerW) -> B (H kerH) (W kerW)',
                        H=K, W=K, kerH=resolutions[1]//resolutions[0])
    codes_b = rearrange(codes_levels[2], 'B (H W) (kerH kerW) -> B (H kerH) (W kerW)',
                        H=K, W=K, kerH=resolutions[2]//resolutions[0])
    return codes_t, codes_m, codes_b


# configurations
set_seed(args.seed)

NUM_CLASSES = args.num_classes  # image net
TOP_RESOLUTION = args.top_resolution
BOT_RESOLUTION = args.bot_resolution

TOP_K = args.top_k
TOP_P = args.top_p
TEMP = args.temperature
T_DECAY = args.temperature_decay
N_SAMPLES = args.batch_size

# construct consecutive directiries for intermediate results
result_path = args.result_path
if not os.path.exists(result_path):
    os.makedirs(result_path, exist_ok=True)

# load model
model_top = load_model(args.model_path)
model_top.eval()

pbar = tqdm(range(0, NUM_CLASSES), total=NUM_CLASSES)

for CLS_IDX in pbar:
    CLS_MAX_SAMPLES = 50000 // NUM_CLASSES  # 50 for ImageNet
    for num_batches in range(0, CLS_MAX_SAMPLES // N_SAMPLES):
        targets = torch.ones(N_SAMPLES, dtype=torch.long) * CLS_IDX

        if (args.code_level == 2):
            temps = [TEMP * (T_DECAY ** i) for i in range(0, args.code_level)]
            codes_t, codes_b = sampling_top(model_top.stage2,
                                            CLS_IDX,
                                            top_k=TOP_K,
                                            top_p=TOP_P,
                                            softmax_temperature=temps,
                                            num_candidates=N_SAMPLES,
                                            top_resolution=TOP_RESOLUTION,
                                            bot_resolution=BOT_RESOLUTION,
                                            model_stage1=model_top.stage1)

            pixels = [model_top.stage1.decode_code(codes_t[i:i+1], codes_b[i:i+1]) for i in range(0, N_SAMPLES)]
            pixels = torch.cat(pixels, dim=0) * 0.5 + 0.5
            pixels = torch.clamp(pixels, 0, 1)

        elif (args.code_level == 3):
            temps = [TEMP * (T_DECAY ** i) for i in range(0, args.code_level)]
            codes = sampling_level3(model_top.stage2,
                                    CLS_IDX,
                                    top_k=TOP_K,
                                    top_p=TOP_P,
                                    softmax_temperatures=temps,
                                    num_candidates=N_SAMPLES,
                                    resolutions=[TOP_RESOLUTION*(2**i) for i in range(0, args.code_level)],
                                    model_stage1=model_top.stage1)

            pixels = [model_top.stage1.decode_code([codes[0][i:i+1],
                                                    codes[1][i:i+1], codes[2][i:i+1]]) for i in range(0, N_SAMPLES)]
            pixels = torch.cat(pixels, dim=0) * 0.5 + 0.5
            pixels = torch.clamp(pixels, 0, 1)

        save_pickle(
            os.path.join(result_path, f'samples_({CLS_IDX+1}_{num_batches}).pkl'),
            pixels.cpu().numpy(),
        )

        np.savez(
            os.path.join(result_path, f'targets_({CLS_IDX+1}_{num_batches}).npz'),
            targets=targets.cpu().numpy(),
        )
