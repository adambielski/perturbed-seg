import torch
import torch.nn as nn
import torch.nn.functional as F

class LayeredRenderer:
    def __init__(self):
        pass

    def __call__(self, bg, fg_mask):
        out = bg
        fg, mask = fg_mask
        out = out * (1. - mask) + fg * mask
        return out

class ModelWrapper(nn.Module):

    def __init__(self, generator, perturber, renderer):
        super().__init__()
        self.generator = generator
        if type(perturber) in (list, tuple) and len(perturber) == 0:
            perturber = None
        self.perturber = perturber
        self.renderer = renderer

    def forward(self, *args, **kwargs):
        X = self.generator(*args, **kwargs)
        if self.perturber is not None:
            perturbed = self.perturber(*X)
        else:
            perturbed = X
        rendered = self.renderer(*perturbed)
        return rendered, perturbed, X
