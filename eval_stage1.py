# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import os
import logging
import math

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader

from hqvae.datasets import ImageNet, CC3M, FFHQ
from hqvae.models import build_model
from hqvae.models import ImageGPT2
from hqvae.utils.utils import set_seed
from hqvae.utils.config2 import get_base_config

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dname', type=str, default='imagenet', help='[imagenet, cc3m, ffhq]')
parser.add_argument('-r', '--result-path', type=str, default='./results.tmp')
parser.add_argument('-i', '--input-res', type=int, default=256)
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('--recon-img', type=str, default='all-codes')
parser.add_argument('--code-usage', action='store_true')
parser.add_argument('--fid', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-full-checkpoint', action='store_true')

args = parser.parse_args()


def create_dataset(name):
    transforms_ = [
        transforms.Resize(args.input_res),
        transforms.CenterCrop(args.input_res),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    transforms_ = transforms.Compose(transforms_)

    if name == 'imagenet':
        dataset = ImageNet(split='val', transform=transforms_)
    elif name == 'cc3m':
        dataset = CC3M(split='val', transform=transforms_)
    elif name == 'ffhq':
        dataset = FFHQ(split='val', transform=transforms_)
    else:
        raise ValueError()

    return dataset

# for multi-level hqvae
def recon_image(m_codes, model):
    xs_rec = model.decode_code(m_codes)
    xs_rec = torch.clamp(xs_rec, -1., 1.)
    xs_rec = (xs_rec + 1.) / 2.
    return xs_rec

@torch.no_grad()
def do_recon(model, xs, cnts):
    outputs = model(xs)
    xs_rec, codes = outputs[0], outputs[-1]
    xs_rec = torch.clamp(xs_rec, -1., 1.)
    xs_rec = (xs_rec + 1.) / 2.
    xs = (xs + 1.) / 2.

    if args.code_usage:
        if isinstance(codes, tuple) or isinstance(codes, list):
            num_code_types = len(cnts)
            for i in range(num_code_types):
                code, cnt = codes[i], cnts[i]
                code, count = torch.unique(code, sorted=True, return_counts=True)
                code = code.view(-1).cpu()
                count = count.view(-1).cpu()
                cnt[code] += count            
        else:
            code, cnt = codes, cnts[0]
            code, count = torch.unique(code, sorted=True, return_counts=True)
            code = code.view(-1).cpu()
            count = count.view(-1).cpu()
            cnt[code] += count
    return xs, xs_rec

@torch.no_grad()
def do_recon_all(model, xs, n_levels):
    if n_levels == 2:
        outputs = model.forward_topbottom(xs)
        xs_rec_all, codes = outputs[0], outputs[-1]

        xs_rec_img = []
        for xs_rec in xs_rec_all:
            xs_rec = torch.clamp(xs_rec, -1., 1.)
            xs_rec = (xs_rec + 1.) / 2.
            xs_rec_img.append(xs_rec)
        xs = (xs + 1.) / 2.

        return xs, xs_rec_img[0], xs_rec_img[1], xs_rec_img[2], codes
    elif n_levels == 3:
        codes = model.get_codes(xs)
        
        # reshape
        B = xs.size(0)
        new_codes = []
        for code in codes:
            K = int(math.sqrt(code.numel()/B))
            code = code.view(B, K, K)
            new_codes.append(code)
        
        codes = new_codes
        xs_rec = recon_image([codes[0], None, None], model)

        xs_rec = torch.clamp(xs_rec, -1., 1.)
        xs_rec = (xs_rec + 1.) / 2.

        xs = (xs + 1.) / 2.  
        return xs, xs_rec, codes

def setup_pretrained_model(model_path):
    config = OmegaConf.load(os.path.join(model_path, "config.yaml"))
    model = build_model(config.stage1.type,
                        config.stage1,
                        config.optimizer)
    last_path = os.path.join(model_path, 'ckpt/last.ckpt')
    ckpt_path = os.path.join(model_path, 'ckpt/state_dict.ckpt')

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
    elif os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location='cpu')['state_dict']

    try:
        model.load_state_dict(ckpt, strict=True)
    except RuntimeError:
        print('Changing parameter names for backward compatibility..')
        ckpt_ = {}
        for k, v in ckpt.items():
            if k.startswith('discriminator'):
                ckpt_[k[14:]] = v
            else:
                ckpt_['generator.'+k] = v
        model.load_state_dict(ckpt_, strict=False)
    print(f'{model_path} successfully restored..')
    return model, config

def setup_pretrained_architecture(result_path, device='cuda'):
    config_path = os.path.join(result_path, 'config.yaml')
    last_path = os.path.join(result_path, 'ckpt/last.ckpt')
    ckpt_path = os.path.join(result_path, 'ckpt/state_dict.ckpt')

    config_base = get_base_config(use_default=False)
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(config_base, config)
    model = ImageGPT2(config)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt, strict=True)

    elif os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location='cpu')['state_dict']
        model.load_state_dict(ckpt, strict=True)

    config.stage1.hparams_aux.bottom_start = 100000000000 # no bypass 
    model_stage1 = build_model(config.stage1.type,
                               config.stage1,
                               config.optimizer)
    model_stage1.generator.load_state_dict(model.stage1.state_dict())

    return model_stage1, config


if __name__ == '__main__':
    set_seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create console handler and set level to info
    ch = logging.FileHandler(os.path.join(args.result_path, 'eval.log'), mode='a')
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")
    )
    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(logging.StreamHandler())

    dataset = create_dataset(name=args.dname)
    loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                        batch_size=args.batch_size, num_workers=16)

    fid = FID().cuda()

    if args.use_full_checkpoint:
        # for checkpoint with the entire architecture with stage 1 and 2 models
        model, config = setup_pretrained_architecture(args.result_path)
    else:
        # for checkpoint with stage 1 model
        model, config = setup_pretrained_model(args.result_path)     

    model.cuda()
    model.eval()

    pbar = tqdm(enumerate(loader), total=len(loader))

    if hasattr(model.generator, 'code_levels'):
        n_levels = model.generator.code_levels
    else:
        n_levels = 2
    cnt_codes = [torch.zeros(config.stage1.n_embed, dtype=torch.int64) for _ in range(n_levels)]  # (code_t, code_b)
    n_samples = 0
    mse_loss = 0
    for it, inputs in pbar:
        xs = inputs[0] if isinstance(inputs, list) else inputs
        xs = xs.cuda()

        if args.recon_img == 'top':
            outputs = do_recon_all(model, xs, n_levels)
            xs = outputs[0]
            xs_rec = outputs[1] # top
        else:
           xs, xs_rec = do_recon(model, xs, cnt_codes)
        mse_loss += F.mse_loss(xs, xs_rec, reduction='sum') / (args.input_res*args.input_res*3)
        n_samples += xs.shape[0]

        if args.fid:
            xs_fid = ((xs * 0.5 + 0.5) * 255.).to(dtype=torch.uint8)
            xs_rec_fid = ((xs_rec * 0.5 + 0.5) * 255.).to(dtype=torch.uint8)

            fid.update(xs_fid, real=True)
            fid.update(xs_rec_fid, real=False)
        pbar.set_description("mse_loss: %.4f" % (mse_loss / n_samples))

    xs = xs.permute(0, 2, 3, 1).cpu().numpy()
    xs_rec = xs_rec.permute(0, 2, 3, 1).cpu().numpy()
    fid_score = fid.compute() if args.fid else 0
    mse_loss /= n_samples
    n_used = [(cnt_code > 0).sum() for cnt_code in cnt_codes]
    print(xs_rec.shape, xs.shape)

    summary = 'model: %s, dataset: %s, #samples: %d, mse_loss: %.4f, rFID: %.4f' % (args.result_path, args.dname, n_samples, mse_loss, fid_score)
    if(n_levels == 2):
        summary += ' #top_codes: %d, #bottom_codes: %d' % (int(n_used[0]), int(n_used[1])
        )
    elif(n_levels > 2):
        summary += '\n'
        for ci, num in enumerate(n_used):
            summary += f'#level {ci} codes:{int(num)}, '
        summary = summary[0:len(summary)-2]
    logger.info(summary)


