import os
import numpy as np
import pdb
from torchvision import datasets, transforms
import torch
import torch.utils.data as data
from loaders.mnist import get_heterogeneous_mnist, get_mnist_split, get_mnist_iid
from loaders.cifar import get_cifar

ROOT_LOCAL = './data'
ROOT_CLUSTER = '/mloraw1/danmoral/data'

def _get_mnist(args, root, n_nodes, batch_size):

    if args.p_label_skew > 0:
        return get_heterogeneous_mnist(root, n_nodes, batch_size, args.p_label_skew)
    if args.data_split:
        return get_mnist_split(root, n_nodes, batch_size)
    else:
        return get_mnist_iid(root, batch_size)


def _get_cifar(args, root, batch_size, fraction):
    
    if args.p_label_skew > 0:
        raise Exception('Heterogeneous CIFAR not supported yet')
    elif args.data_split:
        return get_cifar(args, root, batch_size, iid=False, fraction=fraction, noisy=args.label_noise)
    else:
        return get_cifar(args, root, batch_size, iid=True, fraction=fraction, noisy=args.label_noise)


def get_data(args, batch_size, fraction=-1):
    if args.local_exec:
        root = ROOT_LOCAL
    else:
        root = ROOT_CLUSTER

    if args.dataset == 'mnist':
        return _get_mnist(args, root, args.n_nodes[0], batch_size)
    elif args.dataset in ['cifar10', 'cifar100']:
        return _get_cifar(args, root, batch_size, fraction)
    else:
        raise Exception('Dataset not supported')