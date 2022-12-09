# ------------------------------------------------------------------------------------
# Modified from VQGAN (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import torch
import pytorch_lightning as pl
from typing import Optional, Tuple, List
from omegaconf import OmegaConf
from .discriminator import VQLPIPSWithDiscriminator
from ...optimizers import build_scheduler


class VQGAN(pl.LightningModule):
    def __init__(self,
                 generator: torch.nn.Module,
                 hparams_disc: OmegaConf,
                 hparams_opt: OmegaConf) -> None:
        super().__init__()

        self.generator = generator
        if hparams_disc is not None:
            self.discriminator = VQLPIPSWithDiscriminator(**hparams_disc)
        self._num_opt_steps = 0
        self._hparams_opt = hparams_opt

        if hparams_disc is not None:
            if hparams_disc.residual_l1_weight is None:
                self._residual_l1_weight = 0.0
            else:
                self._residual_l1_weight = hparams_disc.residual_l1_weight
        else:
            self._residual_l1_weight = 0.0

    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.generator(x, self.global_step)

    def forward_topbottom(self,
                          x: torch.FloatTensor
                          ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        return self.generator.forward_topbottom(x, self.global_step)

    def decode_code(self, code: torch.LongTensor) -> torch.FloatTensor:
        return self.generator.decode_code(code)

    def get_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
        return self.generator.get_codes(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        xrec, qloss, _ = self(batch[0])
        if isinstance(qloss, tuple) or isinstance(qloss, list):
            if (self._residual_l1_weight > 0.0):
                resid_l1_loss = qloss[-1]
                self.log("train/resid_l1_loss", resid_l1_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            qloss = sum(qloss[0:len(qloss)-1])

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.discriminator(qloss, batch[0], xrec, optimizer_idx, self.global_step,
                                                     last_layer=self.get_last_layer(), split="train")
            rec_loss = log_dict_ae["train/rec_loss"]
            p_loss = log_dict_ae["train/p_loss"]
            g_loss = log_dict_ae["train/g_loss"]

            self.log("train/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/p_loss", p_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            del log_dict_ae['train/rec_loss']
            del log_dict_ae['train/p_loss']
            del log_dict_ae['train/g_loss']

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            if (self._residual_l1_weight > 0.0):
                aeloss += resid_l1_loss * self._residual_l1_weight
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.discriminator(qloss, batch[0], xrec, optimizer_idx, self.global_step,
                                                         last_layer=self.get_last_layer(), split="train")
            self.log("train/disc_loss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            del log_dict_disc['train/disc_loss']
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        xrec, qloss, _ = self(batch[0])
        if isinstance(qloss, tuple) or isinstance(qloss, list):
            qloss = sum(qloss[0:len(qloss)-1])

        aeloss, log_dict_ae = self.discriminator(qloss, batch[0], xrec, 0, self.global_step,
                                                 last_layer=self.get_last_layer(), split="valid")
        self.log("valid/rec_loss", log_dict_ae["valid/rec_loss"],
                 prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/p_loss", log_dict_ae["valid/p_loss"],
                 prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return log_dict_ae["valid/rec_loss"]

    def configure_optimizers(self):
        assert self._hparams_opt.opt_type == 'adam'

        opt_ae = torch.optim.Adam(self.generator.parameters(),
                                  lr=self._hparams_opt.base_lr,
                                  betas=self._hparams_opt.betas)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=self._hparams_opt.base_lr,
                                    betas=self._hparams_opt.betas)
        lr_sch_ae = {
            'scheduler': build_scheduler(opt_ae,
                                         self._hparams_opt.base_lr,
                                         self._hparams_opt.steps_per_epoch,
                                         self._hparams_opt.max_steps,
                                         self._hparams_opt.warmup_config),
            'name': 'lr-opt_ae'
        }
        lr_sch_disc = {
            'scheduler': build_scheduler(opt_disc,
                                         self._hparams_opt.base_lr,
                                         self._hparams_opt.steps_per_epoch,
                                         self._hparams_opt.max_steps,
                                         self._hparams_opt.warmup_config),
            'name': 'lr-opt_disc'
        }
        return [opt_ae, opt_disc], [lr_sch_ae, lr_sch_disc]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        lr_sched = self.lr_schedulers()[optimizer_idx]

        optimizer.step(closure=optimizer_closure)
        lr_sched.step()
        self.log(f"lr_{optimizer_idx}", lr_sched.get_last_lr()[0],
                 on_step=True, on_epoch=False, prog_bar=True, logger=True)
        if optimizer_idx == 0:
            self._num_opt_steps += 1

    def from_ckpt(self, path: str, strict: bool = True, ignore_keys: Optional[List] = None) -> None:
        ckpt = torch.load(path, map_location='cpu')['state_dict']
        if ignore_keys:
            for k in ignore_keys:
                del ckpt[k]
        self.load_state_dict(ckpt, strict=strict)
        print(f'{path} successfully restored..')

    def get_last_layer(self):
        return self.generator.decoder.conv_out.weight
