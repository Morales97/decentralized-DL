
import os
import numpy as np
from numpy import linalg as LA
import pdb
from torchvision import datasets, transforms
import torch
import torch.utils.data as data


def get_minst_test(config, batch_size, reduce=False, reduce_factor=4):

    testdata = datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    if reduce:  # reduce test dataset to speed up iterative experiments
        testdata_split = data.random_split(testdata, [int(testdata.data.shape[0]/reduce_factor) for _ in range(reduce_factor)])
        testdata = testdata_split[0]
    test_loader = data.DataLoader(
                                testdata, 
                                batch_size=batch_size, 
                                shuffle=True)

    return test_loader

def get_mnist_iid(config, batch_size):
    '''
    Return the full dataset, random sampling with replacement
    '''
    traindata = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
                       )
    sampler = data.RandomSampler(traindata, replacement=True, num_samples=batch_size)   # NOTE I think num_samples is the total amount of samples to be sampled
    train_loader = data.DataLoader(traindata, sampler=sampler, batch_size=batch_size)

    test_loader = get_minst_test(config, batch_size, reduce=config['small_test_set'])

    return train_loader, test_loader

def get_mnist_split(config, batch_size):
    '''
    Split dataset randomly between workers -> breaks IID
    '''
    traindata = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
                       )
    
    # sampler = data.RandomSampler(traindata, replacement=True, num_samples=batch_size)   
    traindata_split = data.random_split(traindata, [int(traindata.data.shape[0] / config['n_nodes']) for _ in range(config['n_nodes'])])
    # train_loader = [data.DataLoader(x, batch_size=batch_size, sampler=sampler) for x in traindata_split]
    train_loader = [data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    test_loader = get_minst_test(config, batch_size, reduce=config['small_test_set'])

    return train_loader, test_loader

def get_mnist(config, batch_size):
    if config['data_split'] == 'yes':
        return get_mnist_split(config, batch_size)
    elif config['data_split'] == 'no':
        return get_mnist_iid(config, batch_size)
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