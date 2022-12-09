# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import argparse
import pytorch_lightning as pl

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from hqvae.datasets import DatasetModule
from hqvae.models import build_model
from hqvae.utils.logger import CustomLogger
from hqvae.utils.config1 import build_config
from hqvae.utils.utils import logging_model_size


parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config-path', type=str, default=None, required=True)
parser.add_argument('-r', '--result-path', type=str, default=None, required=True)
parser.add_argument('-u', '--path-upstream', type=str, default='')

parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--n-nodes', type=int, default=1)
parser.add_argument('--n-gpus', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', action='store_true')


args = parser.parse_args()


def setup_callbacks(config, result_path):
    # Setup callbacks
    ckpt_path = os.path.join(result_path, 'ckpt')
    log_path = os.path.join(result_path, 'log')

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=config.dataset.dataset+"-{epoch:02d}",
        every_n_epochs=config.experiment.save_ckpt_freq,
        save_weights_only=False,
        save_last=True
    )
    logger_tb = TensorBoardLogger(log_path, name="vqgan")
    logger_cu = CustomLogger(config, result_path)
    return checkpoint_callback, logger_tb, logger_cu


if __name__ == '__main__':
    pl.seed_everything(args.seed)

    # Setup
    config, result_path = build_config(args)
    ckpt_callback, logger_tb, logger_cu = setup_callbacks(config, result_path)

    # Build data modules
    dataset = DatasetModule(dataset=config.dataset.dataset,
                            image_resolution=config.dataset.image_resolution,
                            train_batch_size=config.experiment.local_batch_size,
                            valid_batch_size=config.experiment.valid_batch_size,
                            num_workers=16)

    dataset.setup()
    train_dataloader = dataset.train_dataloader()
    valid_dataloader = dataset.valid_dataloader()
    if logger_cu._logger is not None:
        logger_cu._logger.info(f"len(train_dataset) = {len(dataset.trainset)}")
        logger_cu._logger.info(f"len(valid_dataset) = {len(dataset.validset)}")

    # Calculate how many batches are accumulated
    total_gpus = args.n_gpus * args.n_nodes
    assert config.experiment.total_batch_size % total_gpus == 0
    grad_accm_steps = config.experiment.total_batch_size // (config.experiment.local_batch_size * total_gpus)
    config.optimizer.max_steps = len(dataset.trainset) // config.experiment.total_batch_size * config.experiment.epochs
    config.optimizer.steps_per_epoch = len(dataset.trainset) // config.experiment.total_batch_size

    # Build a model
    model = build_model(model_name=config.stage1.type,
                        cfg_stage1=config.stage1,
                        cfg_opt=config.optimizer)
    logging_model_size(model, logger_cu._logger)

    if(len(args.path_upstream) > 0):
        print(os.path.join(args.path_upstream, 'stage1_last.ckpt'))
        model.from_ckpt(os.path.join(args.path_upstream, 'stage1_last.ckpt'), strict=True)

    # Build a trainer
    trainer = pl.Trainer(max_epochs=config.experiment.epochs,
                         accumulate_grad_batches=grad_accm_steps,
                         gradient_clip_val=config.optimizer.grad_clip_norm,
                         precision=16 if config.optimizer.use_amp else 32,
                         callbacks=[ckpt_callback, logger_cu],
                         accelerator="gpu",
                         num_nodes=args.n_nodes,
                         devices=args.n_gpus,
                         strategy=DDPPlugin(ddp_comm_hook=default_hooks.fp16_compress_hook) if
                         config.experiment.fp16_grad_comp else "ddp",
                         logger=logger_tb,
                         log_every_n_steps=10)

    trainer.fit(model, train_dataloader, valid_dataloader)
