import argparse
import random
import math
from functools import partial

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad

from utils import TensorboardLogger, MLFlowLogger, CombinedLogger
from utils import LSUNClass
from data import sample_data, MultiResolutionDataset
from model import SimpleBgFgMask, BgFgMask, BgFgMaskSharedStyle
from stylegan import Discriminator

from perturber import CompositePerturber, RandomShift, BgContrastJitter
from renderer import LayeredRenderer, ModelWrapper
from loss import MaskLoss
from utils import NormalNoiseSampler, RealImageSampler
import mlflow
import os
import sys
import shutil
from copy import deepcopy

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult

def get_first_n_images(loader, n_images):
    n = 0
    samples = []
    iter_ = iter(loader)
    while n < n_images:
        samples.append(next(iter_)[0])
        n += len(samples[-1])
    real_samples = torch.cat(samples, dim=0)[:n_images]
    return real_samples

def log_real_images(loader, logger, i, step, n_images=64):
    real_samples = get_first_n_images(loader, n_images)
    logger.log_images(real_samples, tag='real_samples', step=i, epoch=step)

def save(path, generator, g_running, discriminator, g_optimizer, d_optimizer, alpha, step):
    to_save_dict = {
                    'generator': generator.module.generator.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'generator_running': g_running.generator.state_dict(),
                    'alpha': alpha,
                    'step': step,
    }
    
    torch.save(to_save_dict, path)

def train(args, dataset, generator, g_running, discriminator, mask_loss_fn, logger, log_dir, step=None, gen_every=100):
    if step is None:
        step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution, num_workers=args.num_workers, org_to_crop=args.org_to_crop,
        shuffle=True, drop_last=False
    )
    data_loader = iter(loader)
    
    # log_real_images(loader, logger, 0, step)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr_disc_mult*args.lr.get(resolution, 0.001))

    pbar = tqdm(range(3_000_000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    used_sample = args.used_sample

    perturbed_outputs = generator.module.perturber.perturbs()

    gan_sampler_ = NormalNoiseSampler()
    gan_sampler = lambda bsize: gan_sampler_(b_size, code_size)
    real_samples = get_first_n_images(loader, 64)
    logger.log_images(real_samples, tag='real_samples', step=0, epoch=step)
    gen_i, gen_j = 8, 8
    fixed_noise = [torch.randn(gen_j, code_size).to(device) for _ in range(gen_i)]

    for i in pbar:
        d_optimizer.zero_grad()

        alpha = min(1., 1. / args.phase * (used_sample + 1)) if resolution != args.init_size else 1.

        if used_sample > args.phase * 2:
            step += 1
            save(f'{log_dir}/train_step-{step}.model', generator, g_running, discriminator, g_optimizer, d_optimizer, alpha, step-1)
            
            if step > int(math.log2(args.max_size)) - 2:
                break
            else:
                alpha = 0
                used_sample = 0

            resolution = 4 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution, num_workers=args.num_workers, org_to_crop=args.org_to_crop, drop_last=False, 
            )
            log_real_images(loader, logger, i, step)

            data_loader = iter(loader)
            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr_disc_mult*args.lr.get(resolution, 0.001))
        
        # Discriminator - real images
        try:
            real_image, label = next(data_loader)
        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, label = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.to(device)

        metrics = {}
        if args.loss == 'wgan-gp':
            loss = torch.tensor(0.0, device=device)
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict_mean = real_predict.mean()
            dx = real_predict_mean.item()

            real_predict = real_predict_mean - args.real_penalty * (real_predict ** 2).mean()
            loss -= real_predict
            loss_ = loss.item()
            loss.backward()
            metrics['Dstep_D_x'] = dx
            metrics['lossD_real'] = loss_

        elif args.loss == 'r1':
            raise NotImplementedError

        # Discriminator - fake images
        mixing_range = (-1, -1)
        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, code_size, device=device
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
            if args.same_mixing:
                mixing_range = (random.sample(list(range(step)), 1)[0], 100)
        else:
            gen_in1, gen_in2 = gan_sampler(b_size).to(device), gan_sampler(b_size).to(device)

        fake_image = generator(gen_in1, step=step, alpha=alpha, mixing_range=mixing_range)[0]
        fake_d_input = fake_image
        fake_predict = discriminator(fake_d_input, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predict.backward()
            eps = torch.rand(fake_image.size(0), 1, 1, 1).to(device)
            x_hat = eps * real_image.data + (1 - eps) * fake_d_input.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            grad_loss_val = grad_penalty.item()

            metrics['Dstep_D_Gz'] = fake_predict.item()
            metrics['lossD_fake'] = fake_predict.item()
            metrics['lossD'] = fake_predict.item() - real_predict.item()
            metrics['grad_penalty'] = grad_loss_val
    
        elif args.loss == 'r1':
            raise NotImplementedError

        disc_loss_val = metrics['lossD']
        d_optimizer.step()
        d_optimizer.zero_grad()

        # Generator update
        if i % n_critic == 0:
            g_optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            rendered, perturbed, X = generator(gen_in2, step=step, alpha=alpha, mixing_range=mixing_range)
            fake_d_input = rendered

            fake_predict = discriminator(fake_d_input, step=step, alpha=alpha)
            predict = fake_predict
            predict_mean = predict.mean()

            if args.loss == 'wgan-gp':
                loss -= predict_mean
            elif args.loss == 'r1':
                raise NotImplementedError

            # Mask loss
            mask = perturbed[1][1]
            mask_loss, mask_loss_dict = mask_loss_fn(mask)

            gen_loss_val = loss.item()
            loss += mask_loss

            metrics['Gstep_D_Gz'] = predict_mean.item()
            metrics['lossG'] = loss.item()
            metrics['lossG_fake'] = gen_loss_val
            metrics['min_mask_loss'] = mask_loss_dict['min_mask_loss'].item()
            metrics['bin_loss'] = mask_loss_dict['bin_loss'].item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        logger.log_metrics(metrics, i)

        if i % gen_every == 0:
            g_optimizer.zero_grad()
            generator.eval()
            img_keys = ['rendered', 'bg']
            if perturbed_outputs[0]:
                img_keys.append('bg_perturbed')
            
            img_keys.append(f'mask')
            img_keys.append(f'fg')
            img_keys.append(f'fgmask')
            if perturbed_outputs[1]:
                img_keys.append(f'mask_perturbed')
                img_keys.append(f'fg_perturbed')
                img_keys.append(f'fgmask_perturbed')
            img_keys.extend([ik + '_running' for ik in img_keys])

            img_dict = {img_key: [] for img_key in img_keys}
            with torch.no_grad():
                for fnoise in fixed_noise:
                    rendered, perturbed, X = generator(
                        fnoise, step=step, alpha=alpha
                    )
                    
                    img_dict['rendered'].append(rendered.data.cpu())
                    img_dict['bg'].append(X[0].data.cpu())
                    fg, mask = X[1]
                    
                    img_dict[f'fg'].append(fg.data.cpu())
                    img_dict[f'mask'].append(mask.data.cpu())
                    img_dict[f'fgmask'].append((fg * mask).data.cpu())
                    
                    if perturbed_outputs[0]:
                        img_dict['bg_perturbed'].append(perturbed[0].data.cpu())
                    
                    if perturbed_outputs[1]:
                        fg, mask = perturbed[1]
                        img_dict[f'fg_perturbed'].append(fg.data.cpu())
                        img_dict[f'mask_perturbed'].append(mask.data.cpu())
                        img_dict[f'fgmask_perturbed'].append((fg * mask).data.cpu())

                    rendered, perturbed, X = g_running(
                        fnoise, step=step, alpha=alpha
                    )

                    img_dict['rendered_running'].append(rendered.data.cpu())
                    img_dict['bg_running'].append(X[0].data.cpu())
                    fg, mask = X[1]
                    
                    img_dict[f'fg_running'].append(fg.data.cpu())
                    img_dict[f'mask_running'].append(mask.data.cpu())
                    img_dict[f'fgmask_running'].append((fg * mask).data.cpu())

                for key, imgs in img_dict.items():
                    if len(imgs) == 0:
                        continue
                    range_ = (0., 1.) if key.startswith('mask') else (-1., 1.)
                    logger.log_images(torch.cat(imgs, 0), i, step, key, range=range_)
            generator.train()

        if i % 10000 == 0:
            save(f'{log_dir}/train_step-{step}_{i}.model', generator, g_running, discriminator, g_optimizer, d_optimizer, alpha, step)

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    batch_size = 32

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    """ Data details"""
    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--extra_db', type=str, help='extra db to be concatenated')
    parser.add_argument('-d', '--data', default='folder', type=str, choices=['folder', 'lsun', 'lmdb_resized'], help=('Specify dataset. ' 'Currently Image Folder and LSUN is supported'))
    parser.add_argument('--org_to_crop', default=1., type=float, help='the image will be resized to org_to_crop*image_size, then image_size random crops are taken')
    parser.add_argument('--max_images', default=100000, type=int, help='max number of images')
    parser.add_argument('--num_workers', type=int, default=32)

    """ Training details """
    parser.add_argument('--phase', type=int, default=600_000, help='number of samples used for each training phases')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_disc_mult', default=1, type=float, help='multiplier for discriminator learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--max_batch_size', type=int, default=None, help='overrides sched for some scales')
    parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--same_mixing', action='store_true', default=False)
    parser.add_argument('--used_sample', default=0, type=int)
    parser.add_argument('--n_critic', default=1, type=int)
    parser.add_argument('--loss', type=str, default='wgan-gp', choices=['wgan-gp'], help='class of gan loss')
    parser.add_argument('--real_penalty', default=0.001, type=float)

    """ Model details """
    parser.add_argument('--max_size', default=128, type=int, help='max image size')
    parser.add_argument('--n_masks', default=1, type=int)
    parser.add_argument('--one_generator', default=False, action='store_true')
    parser.add_argument('--common_style', action='store_true', default=False)
    parser.add_argument('--n_mlp', default=8, type=int)
    parser.add_argument('--mlp_mult', default=0.01, type=float)

    """ Mask loss parameters """
    parser.add_argument('--min_mask_coverage', default=0.05, type=float)
    parser.add_argument('--mask_alpha', default=2.0, type=float)
    parser.add_argument('--binarization_alpha', default=2.0, type=float)

    """ Perturbers """
    parser.add_argument('--location_jitter', default=0., type=float, help='location will be jittered by jitter*imagesize')
    parser.add_argument('--bg_contrast_jitter', default=0., type=float)
    
    parser.add_argument('--checkpoint', default=None)


    args = parser.parse_args()

    sample_data = partial(sample_data, resized_db=args.data=='lmdb_resized')

    n_critic = args.n_critic
    print('args parsed')
    
    log_dir = mlflow.get_artifact_uri().replace('file://', '')
    with open(os.path.join(log_dir, 'run.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    device = torch.device("cuda:0")

    ### Set up the StyleGAN model
    n_mlp = args.n_mlp

    if args.one_generator:
        generator = SimpleBgFgMask(code_size, n_mlp)
    elif args.common_style:
        generator = BgFgMaskSharedStyle(code_size, n_mlp)
    else:
        generator = BgFgMask(code_size, n_mlp)

    generator = generator.to(device)
    discriminator = Discriminator(in_channels=3).to(device)
    
    g_running = deepcopy(generator).to(device)
    g_running.train(False)

    discriminator = nn.DataParallel(discriminator)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    print('Nets initialized')

    ### Set up perturbers
    perturbers = []
    if args.location_jitter > 0.:
        location_noise_fn = lambda *size, resolution, device=torch.device('cpu'): 2*args.location_jitter*resolution*torch.rand(*size, device=device)-args.location_jitter*resolution
        perturbers.append(RandomShift(location_noise_fn))
    if args.bg_contrast_jitter > 0.:
        perturbers.append(BgContrastJitter(args.bg_contrast_jitter))
    
    perturber = CompositePerturber(*perturbers) if len(perturbers) > 0 else None
    print('Perturbers initialized: ', ', '.join([p.__class__.__name__ for p in perturbers]))

    ### Set up the renderer
    renderer = LayeredRenderer()

    ### Wrap the generator
    gen_wrapped = ModelWrapper(generator, perturber, renderer)
    g_running_wrapped = ModelWrapper(g_running, None, renderer)

    gen_wrapped = nn.DataParallel(gen_wrapped).to(device)

    ### Set up the optimizers
    param_groups = generator.parameter_groups()
    g_optimizer = optim.Adam(param_groups['generator'], lr=args.lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({
        'params': param_groups['style'],
        'lr': args.lr * args.mlp_mult,
        'mult': args.mlp_mult
    })
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    print('Optimizers prepared')

    ### Load checkpoints
    if args.checkpoint:
        model_dict = torch.load(args.checkpoint)
        gen_wrapped.module.generator.load_state_dict(model_dict['generator'])
        g_running_wrapped.generator.load_state_dict(model_dict['generator_running'])
        discriminator.module.load_state_dict(model_dict['discriminator'])
        g_optimizer.load_state_dict(model_dict['g_optimizer'])
        d_optimizer.load_state_dict(model_dict['d_optimizer'])
        try:
            alpha = model_dict['alpha']
        except:
            alpha = 1.0
        args.used_sample = alpha * args.phase - 1
        step = model_dict['step']
        print('Checkpoint loaded')
    else:
        accumulate(g_running, generator, 0)
        step = None

    ### Set up data
    if args.data == 'folder':
        dataset = datasets.ImageFolder(args.path)

    elif args.data == 'lsun':
        dataset = LSUNClass(args.path, target_transform=lambda x: 0, max_images=args.max_images)
        if args.extra_db:
            dataset2 = LSUNClass(args.extra_db, target_transform=lambda x: 0, max_images=args.max_images)
            dataset = ConcatDataset((dataset, dataset2))
    elif args.data == 'lmdb_resized':
        dataset = MultiResolutionDataset(args.path, None)
    print('Dataset initialized')

    ### Set up training
    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}
        if args.max_batch_size is not None:
            args.batch = {k: min(v, args.max_batch_size) for k, v in args.batch.items()}
    else:
        args.lr = {}
        args.batch = {}
    args.gen_sample = {512: (8, 4), 1024: (4, 2)}
    args.batch_default = 32

    ### Set up mask loss
    mask_loss_fn = MaskLoss(args.min_mask_coverage, args.mask_alpha, args.binarization_alpha)

    ### Set up loggers
    tb_logger = TensorboardLogger(log_dir, frequency=10)
    mf_logger = MLFlowLogger(frequency=100)
    loggers = [tb_logger, mf_logger]
    logger = CombinedLogger(loggers)
    logger.log_params(args)


    ### Backup code
    code_dir = os.path.join(log_dir, 'code')
    os.mkdir(code_dir)
    code_files = [p for p in os.listdir() if p.endswith('.py')]
    for file in code_files:
        shutil.copyfile(file, os.path.join(code_dir, file))

    print('Starting training...')
    train(args, dataset, gen_wrapped, g_running_wrapped, discriminator, mask_loss_fn=mask_loss_fn, logger=logger, log_dir=log_dir, step=step)
