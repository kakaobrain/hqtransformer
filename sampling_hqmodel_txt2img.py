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
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf

from hqvae.models import ImageGPT2
from hqvae.datasets import CC3MTextOnly
from hqvae.utils.utils import set_seed
from hqvae.utils.config2 import get_base_config
from hqvae.utils.sampling import sampling_ihqgpt, sampling_hqtransformer


parser = argparse.ArgumentParser()

parser.add_argument('-r', '--result-path', type=str, required=True)
parser.add_argument('-m', '--model-path', type=str, default='', required=True)

parser.add_argument('--top-k', type=int, default=2048, required=False)
parser.add_argument('--top-p', type=float, default=1.0, required=False)
parser.add_argument('--temperature', type=float, default=1.0, required=False)
parser.add_argument('--temperature-decay', type=float, default=1.0, required=False)

parser.add_argument('--code-level', type=int, default=2, required=False)

parser.add_argument('--batch_size', type=int, default=32, required=False)
parser.add_argument('--top-resolution', type=int, default=8, required=False)
parser.add_argument('--bot-resolution', type=int, default=16, required=False)
parser.add_argument('--seed', type=int, default=0, required=False)
parser.add_argument('--dataset', type=str, default='cc3m', choices=['cc3m'])

args = parser.parse_args()


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
    loader = DataLoader(dataset_val, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=16)
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

    ckpt = torch.load(result_path, map_location='cpu')['state_dict']

    model.load_state_dict(ckpt, strict=True)
    model.to(device=device)

    return model, config


def save_pickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


def sampling_top(model,
                 cond,
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
                                       cond=cond,
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
                    cond,
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
                                          cond=cond,
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

TOP_RESOLUTION = args.top_resolution
BOT_RESOLUTION = args.bot_resolution

num_batches = args.batch_size

TEMP = args.temperature
T_DECAY = args.temperature_decay
temps = [TEMP * (T_DECAY ** i) for i in range(0, args.code_level)]

# construct consecutive directiries for intermediate results
result_path = args.result_path
if not os.path.exists(result_path):
    os.makedirs(result_path)

# load model
model_top, config = load_model(args.model_path)
model_top.eval()

# data loader
loader = get_text_loader(args, config)

for batch_idx, (_, txts) in tqdm(enumerate(loader)):

    if (args.code_level == 2):
        codes_t, codes_b = sampling_top(model_top.stage2,
                                        cond=txts.cuda(),
                                        top_k=args.top_k,
                                        top_p=args.top_p,
                                        softmax_temperature=temps,
                                        num_candidates=1,
                                        top_resolution=TOP_RESOLUTION,
                                        bot_resolution=BOT_RESOLUTION,
                                        model_stage1=model_top.stage1)

        pixels = [model_top.stage1.decode_code(codes_t[i:i+1], codes_b[i:i+1]) for i in range(0, num_batches)]
        pixels = torch.cat(pixels, dim=0) * 0.5 + 0.5
        pixels = torch.clamp(pixels, 0, 1)

    elif (args.code_level == 3):
        codes = sampling_level3(model_top.stage2,
                                cond=txts.cuda(),
                                top_k=args.top_k,
                                top_p=args.top_p,
                                softmax_temperatures=temps,
                                num_candidates=1,
                                resolutions=[TOP_RESOLUTION*(2**i) for i in range(0, args.code_level)],
                                model_stage1=model_top.stage1)

        pixels = [model_top.stage1.decode_code([codes[0][i:i+1],
                                                codes[1][i:i+1], codes[2][i:i+1]]) for i in range(0, num_batches)]
        pixels = torch.cat(pixels, dim=0) * 0.5 + 0.5
        pixels = torch.clamp(pixels, 0, 1)


    save_pickle(
        os.path.join(result_path, f'samples_({batch_idx+1}_{num_batches}).pkl'),
        pixels.cpu().numpy(),
    )
