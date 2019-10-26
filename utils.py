import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import torch
    
import pickle
import torch.utils.data as data
import string
import six
from PIL import Image
import lmdb

class LSUNClass(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, max_images=None, offset=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = '_cache_' + ''.join(c for c in root if c in string.ascii_letters)
        cache_file_2 = '_cache_{}_{}_'.format(offset, max_images) + ''.join(c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
            self.keys = [self.keys[i] for i in range(offset, offset+max_images)]
        elif max_images is not None and os.path.isfile(cache_file_2):
            if os.path.isfile(cache_file_2):
                self.keys = pickle.load(open(cache_file_2, "rb"))
        elif max_images is None:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, 'wb'))
        else:
            ind = 0
            keys = []
            print('Preparing LMDB cache...')
            with self.env.begin(write=False) as txn:
                for key, _ in txn.cursor():
                    if offset > ind:
                        ind += 1
                    else:
                        keys.append(key)
                        ind += 1
                    if len(keys) == max_images:
                        break
            self.keys = keys
            pickle.dump(self.keys, open(cache_file_2, "wb"))
            print('LMDB cache prepared')

        if max_images is not None:
            self.length = max_images

        self.offset = offset

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'

class NormalNoiseSampler:

    def sample(self, batch, dim):
        return torch.randn(batch, dim)

    def __call__(self, batch, dim):
        return self.sample(batch, dim)

class UniformNoiseSampler:

    def __init__(self, min=-1., max=1.):
        self.min = min
        self.max = max

    def sample(self, batch, dim):
        return torch.rand(batch, dim) * (self.max - self.min) + self.min

    def __call__(self, batch, dim):
        return self.sample(batch, dim)

class ImageSampler:
    def __init__(self):
        pass

    def next_batch(self):
        raise NotImplementedError()

class RealImageSampler(ImageSampler):
    def __init__(self, dataloader, with_label=True):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.with_label=with_label

    def _next_batch_(self):
        data = next(self.iterator)
        if not self.with_label:
            data = data[0]
        return data

    def next_batch(self):
        try:
            return self._next_batch_()
        except StopIteration as e:
            self.iterator = iter(self.dataloader)
            return self._next_batch_()


class FakeImageSampler(ImageSampler):
    def __init__(self, netG, batch_size, z_dim, sampler=NormalNoiseSampler()):
        self.netG = netG
        self.sampler = sampler
        self.batch_size = batch_size
        self.z_dim = z_dim

    def next_batch(self):
        z = self.sampler(self.batch_size, self.z_dim).view(self.batch_size, self.z_dim, 1, 1).cuda()
        return self.netG(z)


from mlflow import log_artifact, log_param, log_metric, log_metrics

def log_params(opt):
    for arg in vars(opt):
        v = getattr(opt, arg)
        if v and v != '':
            log_param(arg, v)


from threading import Thread


class LogThread(Thread):
    def __init__(self, metric_dict, step=None):
        Thread.__init__(self)
        self.metric_dict = dict(**metric_dict)
        self.step = step

    def run(self):
        log_metrics(self.metric_dict, self.step)


class LogArtifactThread(Thread):
    def __init__(self, file):
        Thread.__init__(self)
        self.file=file

    def run(self):
        log_artifact(self.file)


def log_file(file):
    LogArtifactThread(file).start()

from torch.utils.tensorboard import SummaryWriter
import os
from torchvision.utils import make_grid, save_image
import mlflow
from math import sqrt
class TensorboardLogger:

    def __init__(self, root_dir, frequency):
        self.tb_dir = os.path.join(root_dir, 'tb')
        self.writer = SummaryWriter(self.tb_dir)
        self.frequency = frequency

    def log_params(self, opt):
        text = ''
        for arg in vars(opt):
            v = getattr(opt, arg)
            if v is not None and v != '':
                text += '--{} {} '.format(arg, v)
        self.writer.add_text('parameters', text)

    def log_metrics(self, metric_dict, step):
        # for k, v in metric_dict.items():
        #     self.writer.add_scalar(k, v, step)
        if step % self.frequency == 0:
            # self.writer.add_scalars(None, metric_dict, step)
            for k, v in metric_dict.items():
                self.writer.add_scalar(k, v, step)

    def log_images(self, images, step=None, epoch=None, tag='fake_images', range=(-1, 1)):
        """images is batch of images"""
        grid = make_grid(images, nrow=int(sqrt(len(images))), padding=2, normalize=True, range=range, scale_each=False)
        self.writer.add_image(tag, grid, step)
        # self.writer.add_image(tag+'_epoch', grid, epoch)

    def log_images_once(self, images, tag, range=(-1, 1)):
        grid = make_grid(images, nrow=int(sqrt(len(images))), padding=2, normalize=True, range=range,
                         scale_each=False)
        self.writer.add_image(tag, grid)

    def log_histogram(self, values, step=None, epoch=None, tag=None):
        self.writer.add_histogram(tag, values, step)



class MLFlowLogger:

    def __init__(self, frequency):
        self.log_dir = mlflow.get_artifact_uri().replace('file://', '')
        self.frequency = frequency

    def log_params(self, opt):
        for arg in vars(opt):
            v = getattr(opt, arg)
            if v is not None and v != '':
                log_param(arg, v)

    def log_metrics(self, metric_dict, step):
        if step % self.frequency == 0:
            LogThread(metric_dict, step).start()

    def log_images(self, images, step, epoch, tag='fake_images', range=(-1, 1)):
        """images is batch of images"""
        fake_samples_file = '{}/{}_epoch_{:03d}.png'.format(self.log_dir, tag, epoch)
        save_image(images.detach(),
                   fake_samples_file,
                   normalize=True,
                   range=range)

    def log_images_once(self, images, tag, range=(-1, 1)):
        fake_samples_file = '{}/{}.png'.format(self.log_dir, tag)
        save_image(images.detach(),
                   fake_samples_file,
                   normalize=True,
                   range=range)

    def log_histogram(self, values, step=None, epoch=None, tag=None):
        pass

# import wandb
# class WBLogger:

#     def __init__(self, frequency, model=None, log_str='all', project='my-project'):
#         wandb.init(project=project)
#         if model is not None:
#             wandb.watch(model, log=log_str, log_freq=frequency)
#         self.frequency = frequency

#     def log_params(self, opt):
#         wandb.config.update(opt)

#     def log_metrics(self, metric_dict, step):
#         if step % self.frequency == 0:
#             wandb.log(metric_dict, step=step)

#     def log_images(self, images, step, epoch, tag='fake_images', range=(-1, 1)):
#         """images is batch of images"""
#         grid = make_grid(images, nrow=int(sqrt(len(images))), padding=2, normalize=True, range=range, scale_each=False)
#         wandb.log({tag: [wandb.Image(grid)]}, step=step)

#     def log_images_once(self, images, tag, range=(-1, 1)):
#         grid = make_grid(images, nrow=int(sqrt(len(images))), padding=2, normalize=True, range=range, scale_each=False)
#         wandb.log({tag: [wandb.Image(grid)]})


class CombinedLogger:
    def __init__(self, loggers):
        self.loggers = loggers

    def log_params(self, opt):
        for logger in self.loggers:
            logger.log_params(opt)

    def log_metrics(self, metric_dict, step):
        for logger in self.loggers:
            logger.log_metrics(metric_dict, step)

    def log_images(self, images, step, epoch, tag, range=(-1, 1)):
        for logger in self.loggers:
            logger.log_images(images, step, epoch, tag, range)

    def log_images_once(self, images, tag, range=(-1, 1)):
        for logger in self.loggers:
            logger.log_images_once(images, tag, range)

    def log_histogram(self, values, step=None, epoch=None, tag=None):
        for logger in self.loggers:
            logger.log_histogram(values, step, epoch, tag)
