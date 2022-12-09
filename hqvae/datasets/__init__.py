# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision

from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from hqvae.tokenizers import create_tokenizer


ROOT_DIR_IMAGENET = ''
ROOT_DIR_CC3M = ''
ROOT_DIR_CC12M = ''
ROOT_DIR_FFHQ = ''

class DatasetModule(pl.LightningDataModule):
    def __init__(self,
                 dataset: str = 'imagenet',
                 image_resolution: int = 256,
                 train_batch_size: int = 2,
                 valid_batch_size: int = 32,
                 num_workers: int = 8,
                 tok_name: str = 'bpe16k_huggingface',
                 context_length: int = 64,
                 bpe_dropout: float = 0.1
                 ):

        super().__init__()
        self.dataset = dataset
        self.image_resolution = image_resolution
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers

        self.tok_name = tok_name
        self.context_length = context_length
        self.bpe_dropout = bpe_dropout

        if ('ffhq' in dataset):
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_resolution, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.valid_transform = transforms.Compose([
                transforms.Resize(image_resolution),
                transforms.CenterCrop(image_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize(image_resolution),
                transforms.RandomCrop(image_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            self.valid_transform = transforms.Compose([
                transforms.Resize(image_resolution),
                transforms.CenterCrop(image_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def setup(self, stage=None):
        if self.dataset == 'imagenet':
            root = ROOT_DIR_IMAGENET
            self.trainset = torchvision.datasets.ImageNet(root=root, split='train', transform=self.train_transform)
            self.validset = torchvision.datasets.ImageNet(root=root, split='val', transform=self.valid_transform)
        elif self.dataset == 'cc15m':
            self.trainset = CC15M(split='train', tok_name=self.tok_name,
                                  context_length=self.context_length,
                                  transform=self.train_transform, dropout=self.bpe_dropout)
            self.validset = CC15M(split='val', tok_name=self.tok_name,
                                  context_length=self.context_length, transform=self.valid_transform)
        elif self.dataset == 'cc3m':
            self.trainset = CC3M(split='train', tok_name=self.tok_name,
                                 context_length=self.context_length, transform=self.train_transform,
                                 dropout=self.bpe_dropout)
            self.validset = CC3M(split='val', tok_name=self.tok_name,
                                 context_length=self.context_length, transform=self.valid_transform)
        elif self.dataset == 'ffhq':
            self.trainset = FFHQ(split='train', transform=self.train_transform)
            self.validset = FFHQ(split='val', transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def valid_dataloader(self):
        return DataLoader(self.validset,
                          batch_size=self.valid_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)


class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, split='train', transform=None):
        super().__init__(root=ROOT_DIR_IMAGENET, split=split, transform=transform)


class CC3M(VisionDataset):
    splits = {'train', 'val'}

    def __init__(self, **kwargs):

        split_ = kwargs['split']
        transforms_ = kwargs['transform']

        if 'tok_name' in kwargs:
            tok_name = kwargs['tok_name']
        else:
            tok_name = 'bpe16k_huggingface'

        if 'context_length' in kwargs:
            context_length = kwargs['context_length']
        else:
            context_length = 32

        if 'dropout' in kwargs:
            dropout = kwargs['dropout']
        else:
            dropout = None

        assert split_ in self.splits, f'{split_} is not in {self.splits}'

        root_dir = ROOT_DIR_CC3M
        super().__init__(root_dir, transform=transforms_)

        self.split = split_

        self.tokenizer = create_tokenizer(tok_name, lowercase=True, dropout=dropout)
        self.context_length = context_length

        self.tokenizer.add_special_tokens(["[PAD]"])
        self.tokenizer.enable_padding(length=self.context_length,
                                      pad_id=self.tokenizer.token_to_id("[PAD]"))
        self.tokenizer.enable_truncation(max_length=self.context_length)

        self.items = []
        for line in open(f'{self.root}/{split_}_list.txt', 'r').readlines():
            toks = line.strip().split('\t')
            assert len(toks) == 2
            (imgpath, text) = toks
            self.items.append((os.path.join(self.root, imgpath), text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        imgpath, text = self.items[item]

        output = self.tokenizer.encode(text)
        ids = output.ids
        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)

        img = Image.open(imgpath).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, ids


class CC3MTextOnly(CC3M):

    def __getitem__(self, item):
        _, text = self.items[item]

        output = self.tokenizer.encode(text)
        ids = output.ids
        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)

        return 0, ids


class CC15M(VisionDataset):
    splits = {'train', 'val'}

    def __init__(self, split, tok_name, transform=None, context_length=64, dropout=None,
                 cc12m_root = ROOT_DIR_CC12M,
                 cc3m_root = ROOT_DIR_CC3M
                 ):
        assert split in self.splits, f'{split} is not in {self.splits}'
        super().__init__(cc12m_root, transform=transform)


        self.split = split
        self.tokenizer = create_tokenizer(tok_name, lowercase=True, dropout=dropout)
        self.context_length = context_length

        self.tokenizer.add_special_tokens(["[PAD]"])
        self.tokenizer.enable_padding(length=self.context_length,
                                      pad_id=self.tokenizer.token_to_id("[PAD]"))
        self.tokenizer.enable_truncation(max_length=self.context_length)

        self.items = []

        if split == 'train':
            list_names = [
                f'{cc3m_root}/train_list.txt',
                f'{cc12m_root}/cc12m_with_hash_no_url_only_valid.tsv'
            ]
        else:
            list_names = [f'{cc3m_root}/val_list.txt']

        for idx, list_name in enumerate(list_names):
            for line in open(list_name, 'r').readlines():
                toks = line.strip().split('\t')
                assert len(toks) == 2
                (imgpath, text) = toks
                if split == 'train':
                    if idx == 0:
                        self.items.append((os.path.join(cc3m_root, imgpath), text))
                    else:
                        self.items.append((os.path.join(cc12m_root, 'images', f'{imgpath}.jpg'), text))
                else:
                    self.items.append((os.path.join(cc3m_root, imgpath), text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        imgpath, text = self.items[item]

        img = Image.open(imgpath).convert('RGB')
        if self.transform:
            img = self.transform(img)

        output = self.tokenizer.encode(text)
        ids = output.ids
        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)

        return img, ids


class ImageFolder(VisionDataset):
    def __init__(self, root, train_list_file, val_list_file, split='train', **kwargs):
        super().__init__(root, **kwargs)

        self.train_list_file = train_list_file
        self.val_list_file = val_list_file

        self.split = self._verify_split(split)

        self.loader = torchvision.datasets.folder.default_loader
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

        if self.split == 'trainval':
            fname_list = os.listdir(self.root)
            samples = [self.root.joinpath(fname) for fname in fname_list
                       if fname.lower().endswith(self.extensions)]
        else:
            listfile = self.train_list_file if self.split == 'train' else self.val_list_file
            with open(listfile, 'r') as f:
                samples = [self.root.joinpath(line.strip()) for line in f.readlines()]

        self.samples = samples

    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def valid_splits(self):
        return 'train', 'val', 'trainval'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index, with_transform=True):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transforms is not None and with_transform:
            sample, _ = self.transforms(sample, None)
        return sample, 0


class FFHQ(ImageFolder):
    root = ROOT_DIR_FFHQ
    train_list_file = Path(__file__).parent.joinpath('assets/ffhqtrain.txt')
    val_list_file = Path(__file__).parent.joinpath('assets/ffhqvalidation.txt')

    def __init__(self, split='train', **kwargs):
        super().__init__(FFHQ.root, FFHQ.train_list_file, FFHQ.val_list_file, split, **kwargs)
