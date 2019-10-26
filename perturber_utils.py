import torch

def contrast_jitter(x: torch.Tensor, cj: torch.Tensor):
    mean = x.mean(dim=(2, 3), keepdim=True) # Nx3x1x1
    cj = cj.view(mean.size(0), 1, 1, 1)
    
    x_min = x.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    x_max = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    
    x_ = (x-mean) * cj + mean
    return torch.min(x_max, torch.max(x_min, x_))

def random_contrast_jitter(x: torch.Tensor, strength=0.1):
    cj = 1 + (2*strength*torch.rand(x.size(0), 1, 1, 1, device=x.device) - strength)
    return contrast_jitter(x, cj)

def shift(x_group, shifts:torch.Tensor):
    x_outs = []
    for x in x_group:
        x_outs.append(torch.zeros_like(x, device=x.device))

    N, _, H, W = x_group[0].size()

    for i in range(N):
        for x_out, x_org in zip(x_outs, x_group):
            x_out[i, :, max(0, shifts[i][1]):min(H, shifts[i][1] + H), max(0, shifts[i][0]):min(W, shifts[i][0] + W)] = x_org[i, :, max(0, -shifts[i][1]):min(H, -shifts[i][1] + H), max(0, -shifts[i][0]):min(W, -shifts[i][0] + W)]

    return x_outs
