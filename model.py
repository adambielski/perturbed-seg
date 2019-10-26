import torch
import torch.nn as nn

from stylegan import Generator, GeneratorNOutputs, StyledGenerator, StyledGenerators

def get_masks(fg_masks):
    return tuple([fm[1] for fm in fg_masks])

class SimpleBgFgMask(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super(SimpleBgFgMask, self).__init__()

        output_dims = [3, 3, 1]
        self.generator = GeneratorNOutputs(code_dim, output_dims=output_dims)
        self.generator = StyledGenerator(self.generator, code_dim, n_mlp, code_dim)

    def forward(self, x, noise=None, step=0, alpha=-1, mean_style=None, style_weight=0, mixing_range=(-1, -1)):
        bg, fg, mask = self.generator(x, noise, step, alpha, mean_style, style_weight,  mixing_range)
        mask = torch.sigmoid(mask)
        return bg, (fg, mask)

    def parameter_groups(self):
        groups = {}
        groups['style'] = self.generator.style.parameters()
        groups['generator'] = self.generator.generator.parameters()
        return groups

class BgFgMask(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super(BgFgMask, self).__init__()

        output_dims = [3, 1]
        self.generator_bg = StyledGenerator(Generator(code_dim), code_dim, n_mlp, code_dim)
        self.generator_objects = StyledGenerator(GeneratorNOutputs(code_dim, output_dims=output_dims), code_dim, n_mlp, code_dim)

    def forward(self, x, noise=None, step=0, alpha=-1, mean_style=None, style_weight=0, mixing_range=(-1, -1)):
        bg = self.generator_bg(x, noise, step, alpha, mean_style, style_weight,  mixing_range)
        fg, mask = self.generator_objects(x, noise, step, alpha, mean_style, style_weight,  mixing_range)
        mask = torch.sigmoid(mask)
        return bg, (fg, mask)

    def parameter_groups(self):
        groups = {}
        groups['style'] = list(self.generator_bg.style.parameters()) + list(self.generator_objects.style.parameters())
        groups['generator'] = list(self.generator_bg.generator.parameters()) + list(self.generator_objects.generator.parameters())
        return groups

class BgFgMaskSharedStyle(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super(BgFgMaskSharedStyle, self).__init__()

        output_dims = [3, 1]
        generator_bg = Generator(code_dim)
        generator_objects = GeneratorNOutputs(code_dim, output_dims=output_dims)
        self.generator = StyledGenerators((generator_bg, generator_objects), code_dim, n_mlp, code_dim)

    def forward(self, x, noise=None, step=0, alpha=-1, mean_style=None, style_weight=0, mixing_range=(-1, -1)):
        bg, out = self.generator(x, noise, step, alpha, mean_style, style_weight,  mixing_range=mixing_range)
        fg, mask = out[0], torch.sigmoid(out[1])
        return bg, (fg, mask)

    def parameter_groups(self):
        groups = {}
        groups['style'] = self.generator.style.parameters()
        groups['generator'] = []
        for gen in self.generator.generators:
            groups['generator'] += list(gen.parameters())
        return groups
