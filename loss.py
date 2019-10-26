import torch
import torch.nn.functional as F
from functools import partial

def min_mask_loss(mask, min_mask_coverage):
    return F.relu(min_mask_coverage - mask.mean(dim=(1, 2, 3))).mean()

def min_permask_loss(mask, min_mask_coverage):
    '''
    One object mask per channel in this case
    '''
    return F.relu(min_mask_coverage - mask.mean(dim=(2, 3))).mean()

def min_mask_loss_batch(mask, min_mask_coverage):
    return F.relu(min_mask_coverage - mask.mean())

def binarization_loss(mask):
    return torch.min(1-mask, mask).mean()


class MaskLoss:
    def __init__(self, min_mask_coverage, mask_alpha, bin_alpha, min_mask_fn=min_permask_loss):
        self.min_mask_coverage = min_mask_coverage
        self.mask_alpha = mask_alpha
        self.bin_alpha = bin_alpha
        self.min_mask_fn = partial(min_mask_fn, min_mask_coverage=min_mask_coverage)

    def __call__(self, mask):
        if type(mask) in (tuple, list):
            mask = torch.cat(mask, dim=1)
        min_loss = self.min_mask_fn(mask)
        bin_loss = binarization_loss(mask)
        return self.mask_alpha * min_loss + self.bin_alpha * bin_loss, dict(min_mask_loss=min_loss, bin_loss=bin_loss)