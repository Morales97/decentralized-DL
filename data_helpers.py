
import os
import numpy as np
from numpy import linalg as LA
import pdb
from torchvision import datasets, transforms
import torch
import torch.utils.data as data


def get_mnist_iid(config):
    '''
    Return the full dataset, random sampling with replacement
    '''
    traindata = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
                       )
    sampler = data.RandomSampler(traindata, replacement=True, num_samples=config['batch_size'])   # NOTE I think num_samples is the total amount of samples to be sampled
    train_loader = data.DataLoader(traindata, sampler=sampler, batch_size=config['batch_size'])

    test_loader = data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            ), batch_size=10*config['batch_size'], shuffle=True)

    return train_loader, test_loader

def get_mnist_split(config):
    '''
    Split dataset randomly between workers -> breaks IID
    '''
    traindata = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
                       )
    
    # sampler = data.RandomSampler(traindata, replacement=True, num_samples=config['batch_size'])   
    traindata_split = data.random_split(traindata, [int(traindata.data.shape[0] / config['n_nodes']) for _ in range(config['n_nodes'])])
    # train_loader = [data.DataLoader(x, batch_size=config['batch_size'], sampler=sampler) for x in traindata_split]
    train_loader = [data.DataLoader(x, batch_size=config['batch_size'], shuffle=True) for x in traindata_split]

    test_loader = data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            ), batch_size=10*config['batch_size'], shuffle=True)

    return train_loader, test_loader

def get_mnist(config):
    if config['data_split'] == 'yes':
        return get_mnist_split(config)
    elif config['data_split'] == 'no':
        return get_mnist_iid(config)
    else:
        raise Exception('data split modality not supported')

def get_next_batch(config, train_loader, i):
    '''
    Sample a batch of MNIST samples. Supports data split or sampling from full dataset
    '''
    if config['data_split'] == 'yes':
        pdb.set_trace()
        input, target = next(iter(train_loader[i]))
    elif config['data_split'] == 'no':
        pdb.set_trace()
        input, target = next(iter(train_loader))

    return input, target