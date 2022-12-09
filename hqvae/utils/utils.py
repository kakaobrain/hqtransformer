import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def logging_model_size(model, logger):
    if logger is None:
        return
    logger.info(
        "[OPTION: ALL] #params: %.4fM", sum(p.numel() for p in model.parameters()) / 1e6
    )
    logger.info(
        "[OPTION: Trainable] #params: %.4fM", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    )
