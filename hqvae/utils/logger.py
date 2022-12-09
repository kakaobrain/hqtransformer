# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import logging
import torch
import torchvision

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import OmegaConf


class CustomLogger(Callback):
    def __init__(self, config, result_path, is_eval=False):
        super().__init__()

        self._config = config
        self._result_path = result_path
        self._logger = self._init_logger(is_eval=is_eval)

    @rank_zero_only
    def _init_logger(self, is_eval=False):
        self.save_config()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # create console handler and set level to info
        ch = logging.FileHandler(os.path.join(self._result_path, 'eval.log' if is_eval else 'train.log'))
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S")
        )
        # add ch to logger
        logger.addHandler(ch)
        logger.info(f"Logs will be recorded in {self._result_path}...")
        return logger

    @rank_zero_only
    def save_config(self):
        if not os.path.exists(self._result_path):
            os.makedirs(self._result_path)
        with open(os.path.join(self._result_path, 'config.yaml'), 'w') as fp:
            OmegaConf.save(config=self._config, f=fp)

    @rank_zero_only
    def log_img(self, pl_module, batch, global_step, split="train"):
        with torch.no_grad():
            images, labels = batch
            recons = pl_module(images[:16])[0]
            images = images.cpu()

            if(isinstance(recons, tuple)):
                top_recons = recons[0].cpu()
                recons = recons[1].cpu()
            else:
                top_recons = None
                recons = recons.cpu()

            grid_org = (torchvision.utils.make_grid(images, nrow=4) + 1.0) / 2.0
            grid_rec = (torchvision.utils.make_grid(recons, nrow=4) + 1.0) / 2.0
            grid_rec = torch.clip(grid_rec, min=0, max=1)

            pl_module.logger.experiment.add_image(f"images_org/{split}", grid_org, global_step=global_step)
            pl_module.logger.experiment.add_image(f"images_rec/{split}", grid_rec, global_step=global_step)

            if top_recons is not None:
                grid_top = (torchvision.utils.make_grid(top_recons, nrow=4) + 1.0) / 2.0
                grid_top = torch.clip(grid_top, min=0, max=1)
                pl_module.logger.experiment.add_image(f"images_top/{split}", grid_top, global_step=global_step)

    @rank_zero_only
    def log_metrics(self, trainer, split='valid'):
        metrics = []
        for k, v in trainer.callback_metrics.items():
            if split == 'valid':
                if k.startswith('valid'):
                    k = k.split('/')[-1].strip()
                    metrics.append((k, v))
            else:
                if k.startswith('train') and k.endswith('epoch'):
                    k = k.split('/')[-1].strip()[:-6]
                    metrics.append((k, v))
        metrics = sorted(metrics, key=lambda x: x[0])
        line = ','.join([f"  {metric[0]}:{metric[1].item():.4f}" for metric in metrics])
        line = f'EPOCH:{trainer.current_epoch}, {split.upper()}\t' + line
        self._logger.info(line)

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, split='train')

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, split='valid')

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module.discriminator.perceptual_loss.eval()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module._num_opt_steps % self._config.experiment.img_logging_freq == 0:
            pl_module.eval()
            self.log_img(pl_module, batch, global_step=pl_module._num_opt_steps, split="train")
            pl_module.train()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            pl_module.eval()
            self.log_img(pl_module, batch, global_step=trainer.current_epoch, split="valid")
