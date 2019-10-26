import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
from utils import LSUNClass
import math

def resize_and_convert(img, size, quality=100):
    img = trans_fn.resize(img, (size, size))
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()
    return val

def resize_multiple(img, sizes=(8, 16, 32, 64, 128, 256, 512, 1024), quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, quality))

    return imgs

def resize_worker(index, dataset, sizes):
    # i, file = img_file
    img = dataset[index][0]
    out = resize_multiple(img, sizes=sizes)
    return index, out

class ResizeWorker:
    def __init__(self,dataset,sizes):
        self.dataset = dataset
        self.sizes = sizes

    def __call__(self, index):
        img = self.dataset[index][0]
        out = resize_multiple(img, sizes=self.sizes)
        return index, out

def prepare(transaction, dataset, n_worker, sizes=(8, 16, 32, 64, 128, 256, 512, 1024)):
    resize_fn = partial(resize_worker, dataset=dataset, sizes=sizes)
    indices = list(range(len(dataset)))
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i in tqdm(indices):
            i, imgs = resize_fn(i)
            for size, img in zip(sizes, imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                txn.put(key, img)
            total += 1
        txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('path', type=str)
    parser.add_argument('--max_images', default=100000, type=int)
    parser.add_argument('--max_size', default=128, type=int)
    parser.add_argument('--org_to_crop', default=1.0, type=float)

    args = parser.parse_args()

    imgset = LSUNClass(args.path, max_images=args.max_images)

    sizes = [8,16,32,64,128,256,512,1024][:int(math.log2(args.max_size))-2]
    if args.org_to_crop != 1.0:
        sizes = [int(args.org_to_crop*size) for size in sizes]

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, imgset, args.n_worker, sizes=sizes)