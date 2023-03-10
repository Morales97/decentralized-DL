import os
from re import A
import numpy as np
import pdb
from torchvision import datasets, transforms
import torch
import torch.utils.data as data
from loaders.mnist import get_heterogeneous_mnist, get_mnist_split, get_mnist_iid
from loaders.cifar import get_cifar, get_cifar_filtered_samples, get_cifar_test
from loaders.tiny_imagenet import get_tinyimagenet

ROOT_LOCAL = './data'
ROOT_CLUSTER = '/mloraw1/danmoral/data'

def _get_mnist(args, root, n_nodes, batch_size):

    if args.p_label_skew > 0:
        return get_heterogeneous_mnist(root, n_nodes, batch_size, args.p_label_skew)
    if args.data_split:
        return get_mnist_split(root, n_nodes, batch_size)
    else:
        return get_mnist_iid(root, batch_size)


def _get_cifar(args, root, batch_size, val_fraction, fraction):
    
    if args.p_label_skew > 0:
        raise Exception('Heterogeneous CIFAR not supported yet')

    elif args.select_samples != '':
        select_samples = np.load(os.path.join(root, args.select_samples + '.npy'))
        return get_cifar_filtered_samples(args, root, None, select_samples)
    elif args.data_split:
        return get_cifar(args, root, batch_size, val_fraction=val_fraction, iid=False, fraction=fraction, noisy=args.label_noise)
    else:
        return get_cifar(args, root, batch_size, iid=True, fraction=fraction, noisy=args.label_noise)


def _get_tiny_imagenet(args, root, batch_size):
    return get_tinyimagenet(args, root, batch_size)


def get_data(args, batch_size, fraction=-1, val_fraction=0, test_transforms=None):
    if args.local_exec:
        root = ROOT_LOCAL
    else:
        root = ROOT_CLUSTER

    if args.dataset == 'mnist':
        return _get_mnist(args, root, args.n_nodes[0], batch_size)
    elif args.dataset in ['cifar10', 'cifar100']:
        if test_transforms:
            return get_cifar_test(args, root, test_transforms)
        return _get_cifar(args, root, batch_size, val_fraction,fraction)
    elif args.dataset == 'tiny-in':
        return _get_tiny_imagenet(args, root, batch_size)
    else:
        raise Exception('Dataset not supported')