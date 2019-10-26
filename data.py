import math
from functools import partial

import torch
import torch.utils.data as data
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from copy import deepcopy

import torch.nn.functional as Ftensor

import lmdb
import os
from io import BytesIO
from PIL import Image
import pickle
import torch.utils.data as data
import string
import six

class DsWrapper(data.Dataset):
    def __init__(self, dataset, transform):
        dataset.transform = None
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return self.dataset.__class__.__name__ + ' (' + self.dataset.root + ')'

class DsResizedWrapper(data.Dataset):
    def __init__(self, dataset, transform, resolution):
        dataset.transform = None
        self.dataset = dataset
        self.transform = transform
        self.resolution = resolution

    def __getitem__(self, index):
        img, target = self.dataset.get(index, self.resolution)
        return self.transform(img), target

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return self.dataset.__class__.__name__ + ' (' + self.dataset.root + ')'

def sample_data(dataset, batch_size, image_size=4, num_workers=16, org_to_crop=1., center_crop=False, shuffle=True, drop_last=False, resized_db=False):
    transform = []
    if not resized_db:
        transform.append(transforms.Resize((int(org_to_crop*image_size), int(org_to_crop*image_size))))
    else:
        dataset.resolution = int(org_to_crop*image_size)
    if org_to_crop > 1.:
        transform.append(transforms.RandomCrop(image_size) if not center_crop else transforms.CenterCrop(image_size))
    transform.extend([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose(transform)

    dataset = DsWrapper(dataset, transform) if not resized_db else DsResizedWrapper(dataset, transform, int(org_to_crop*image_size))

    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)
    return loader


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0

    def get(self, index, resolution):
        with self.env.begin(write=False) as txn:
            key = f'{resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0

