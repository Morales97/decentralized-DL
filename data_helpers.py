
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
    
    traindata_split = data.random_split(traindata, [int(traindata.data.shape[0] / config['n_nodes']) for _ in range(config['n_nodes'])])
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


def get_heterogeneous_mnist(n_nodes, batch_size, p=0.2):
    pass
'''
    # NOTE the idea here is to (1) split data in shards and give two shards per node (full non-IID), get this running
    # and (2) make a pre split of the dataset, p to divide in shards and 1-p to divide IID

    num_nonidd_samples = int(60000 * p // n_nodes)
    num_iid_samples = int(60000*(1-p)//n_nodes)

    traindata = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
                       )
    target_labels = torch.stack([traindata.targets == i for i in range(10)])
    target_labels_sorted = []
    for i in range(n_nodes):
        target_labels_sorted += [torch.where(target_labels[i:i+1].sum(0))[0]]
    




    pdb.set_trace()
    iid_split = []
    non_iid_split = []
    indices_idd = np.zeros(10)
    indices_non_idd = np.zeros(10)
    
    # non-idd
    for i in range(n_nodes):
        n_classes = 2  # shard from 2 classes
        samples_per_class = int(num_nonidd_samples // n_classes)
        classes = np.random.choice(10, n_classes, replace=False)
        for _class in classes:
            idx = indices_non_idd[_class]
            non_iid_split += [torch.where(target_labels[_class].sum(0))[0][idx : idx + num_nonidd_samples]]
            indices_non_idd[_class] += num_noniid_samples
    
    #for i in range(5):
    #    target_labels_split += [torch.where(target_labels[(2 * i):(2 * (i + 1))].sum(0))[0][:num_nonidd_samples_per_client]]
    traindata_split_noniid = [data.Subset(traindata, tl) for tl in target_labels_split]

    # split into IID vs non-IID sets
    split_iid_non_idd = data.random_split(traindata, [num_iid_samples*n_nodes, 60000-num_iid_samples*n_nodes])

    traindata_split_iid = data.random_split(split_iid_non_idd[0], [num_iid_samples for _ in range(n_nodes)])
    
    train_loader = [data.DataLoader(data.ConcatDataset([x,z]), batch_size=batch_size, shuffle=True) for x, z in zip(traindata_split_iid, traindata_split_noniid)]

    test_loader = data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            ), batch_size=10*batch_size, shuffle=True)

    return train_loader, test_loader
'''