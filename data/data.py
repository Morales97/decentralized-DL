import os
import numpy as np
import pdb
from torchvision import datasets, transforms
import torch
import torch.utils.data as data
from mnist import get_heterogeneous_mnist, get_mnist_split, get_mnist_iid
from cifar import get_cifar_split

ROOT_LOCAL = '.'
ROOT_CLUSTER = '/mloraw1/danmoral/data'

def _get_mnist(config, root, n_nodes, batch_size):

    if config['p_label_skew'] > 0:
        return get_heterogeneous_mnist(root, n_nodes, batch_size, config['p_label_skew'])
    if config['data_split'] == 'yes':
        return get_mnist_split(root, n_nodes, batch_size)
    elif config['data_split'] == 'no':
        return get_mnist_iid(root, batch_size)
    else:
        raise Exception('data split modality not supported')


def _get_cifar(config, root, n_nodes, batch_size):
    
    if config['p_label_skew'] > 0:
        raise Exception('Heterogeneous CIFAR not supported yet')
    elif config['data_split'] == 'yes':
        return get_cifar_split(root, n_nodes, batch_size)
    elif config['data_split'] == 'no':
        raise Exception('IID CIFAR not supported yet')


def get_data(config, n_nodes, batch_size, local_exec=False):
    if local_exec:
        root = ROOT_LOCAL
    else:
        root = ROOT_CLUSTER

    if config['dataset'] == 'mnist':
        return _get_mnist(config, root, n_nodes, batch_size)
    elif config['dataset'] in ['cifar10', 'cifar100']:
        return _get_cifar(config, root, n_nodes, batch_size)
    else:
        raise Exception('Dataset not supported')