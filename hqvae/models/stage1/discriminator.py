# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.lpips import LPIPS
from .modules.layers import NLayerDiscriminator
from .modules.utils import weights_init


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, residual_l1_weight=0.0, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", norm_type='bn',
                 use_recon_top=True, use_perceptual_top=False, use_adversarial_top=False):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert norm_type in ["bn", "gn", "actnorm"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 ndf=disc_ndf,
                                                 norm_type=norm_type
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.norm_type = norm_type

        # reconstruction loss for 'top code only'
        self.use_recon_top = use_recon_top
        self.use_perceptual_top = use_perceptual_top
        self.use_adversarial_top = use_adversarial_top

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        if isinstance(reconstructions, tuple):
            rec_t, rec_tb = reconstructions

            if self.use_recon_top:
                rec_t_loss = F.mse_loss(inputs.contiguous(), rec_t.contiguous())
                rec_tb_loss = F.mse_loss(inputs.contiguous(), rec_tb.contiguous())
                rec_loss = (rec_t_loss + rec_tb_loss) / 2
            else:
                rec_loss = F.mse_loss(inputs.contiguous(), rec_tb.contiguous())

            if self.perceptual_weight > 0:
                if self.use_perceptual_top:
                    p_loss_tb = self.perceptual_loss(inputs.contiguous(), rec_tb.contiguous())
                    p_loss_t = self.perceptual_loss(inputs.contiguous(), rec_t.contiguous())
                    p_loss = (p_loss_t + p_loss_tb)/2
                    rec_loss = rec_loss + self.perceptual_weight * p_loss

                else:
                    p_loss = self.perceptual_loss(inputs.contiguous(), rec_tb.contiguous())
                    rec_loss = rec_loss + self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor([0.0])

            nll_loss = rec_loss
            nll_loss = torch.mean(nll_loss)

        else:
            rec_loss = F.mse_loss(inputs.contiguous(), reconstructions.contiguous())
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor([0.0])

            nll_loss = rec_loss
            nll_loss = torch.mean(nll_loss)

        use_tuple_disc = False
        if isinstance(reconstructions, tuple):
            if self.use_adversarial_top:
                use_tuple_disc = True
            else:
                reconstructions = reconstructions[1]

        # now the GAN part
        if optimizer_idx == 0:
            if (use_tuple_disc):
                g_loss_t, logits_fake_t = self.forward_logits_fake(reconstructions[0], cond)
                g_loss_tb, logits_fake_tb = self.forward_logits_fake(reconstructions[1], cond)

                g_loss = (g_loss_t + g_loss_tb)/2
                logits_fake = (logits_fake_t + logits_fake_tb)/2
            else:
                g_loss, logits_fake = self.forward_logits_fake(reconstructions, cond)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            if (use_tuple_disc):
                d_loss_t, logits_real_t, logits_fake_t = self.forward_logits_real_fake(inputs,
                                                                                       reconstructions[0],
                                                                                       cond,
                                                                                       global_step)
                d_loss_tb, logits_real_tb, logits_fake_tb = self.forward_logits_real_fake(inputs,
                                                                                          reconstructions[1],
                                                                                          cond,
                                                                                          global_step)

                d_loss = (d_loss_t + d_loss_tb) / 2
                logits_real = (logits_real_t + logits_real_tb) / 2
                logits_fake = (logits_fake_t + logits_fake_tb) / 2
            else:
                d_loss, logits_real, logits_fake = self.forward_logits_real_fake(inputs,
                                                                                 reconstructions,
                                                                                 cond,
                                                                                 global_step)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

    def forward_logits_fake(self, reconstructions, cond):
        # generator update
        if cond is None:
            assert not self.disc_conditional
            logits_fake = self.discriminator(reconstructions.contiguous())
        else:
            assert self.disc_conditional
            logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))

        g_loss = -torch.mean(logits_fake)
        return g_loss, logits_fake

    def forward_logits_real_fake(self, inputs, reconstructions, cond, global_step):
        # second pass for discriminator update
        if cond is None:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
        else:
            logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
            logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        return d_loss, logits_real, logits_fake
