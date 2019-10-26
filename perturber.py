from perturber_utils import shift, random_contrast_jitter
import math
import numpy as np
import torch
import torch.nn.functional as F

class AbstractPerturber(object):
    def __init__(self):
        pass

    def perturb(self, *input):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        return self.perturb(*input, **kwargs)

    def perturbs(self):
        ''' Returns array of bool indicating which arguments are perturbed '''
        raise NotImplementedError


class CompositePerturber(AbstractPerturber):

    def __init__(self, *perturbers):
        self.perturbers = perturbers

    def perturb(self, *input):
        for perturber in self.perturbers:
            input = perturber(*input)
        return input

    def perturbs(self):
        perturbed = np.array([p.perturbs() for p in self.perturbers])
        return np.any(perturbed, axis=0)

    def __len__(self):
        return len(self.perturbers)

class LayeredPerturber(AbstractPerturber):
    
    def perturb(self, bg, fg_mask):
        raise NotImplementedError


class RandomShift(LayeredPerturber):
    def __init__(self, location_noise_fn):
        self.location_noise_fn = location_noise_fn

    def perturb(self, bg, fg_mask):
        shifts = torch.round(self.location_noise_fn((bg.size(0), 2), device=bg.device, resolution=bg.size(2))).long()
        fg_mask = shift(fg_mask, shifts)
        return bg, fg_mask

    def perturbs(self):
        return (False, True)


class BgContrastJitter(LayeredPerturber):
    def __init__(self, jitter_strength=0.3):
        self.jitter_strength = jitter_strength

    def perturb(self, bg, fg_mask):
        bg = random_contrast_jitter(bg, self.jitter_strength)
        return bg, fg_mask

    def perturbs(self):
        return (True, False)
