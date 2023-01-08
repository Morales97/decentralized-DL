
import os
import numpy as np
import pdb
from torchvision import datasets, transforms
import torch
import torch.utils.data as data


def get_cifar_test(config, root):

    # decide normalize parameter.
    if config['dataset'] == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    elif config['dataset'] == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )

    transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    dataset_loader(
        root=root,
        train=False,
        transform=transform,
        download=True,
    )

    return data.DataLoader(dataset_loader, batch_size=100, shuffle=False)

def get_cifar_split(config, root, n_nodes, batch_size):

    # decide normalize parameter.
    if config['dataset'] == "cifar10":
        dataset = datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    elif config['dataset'] == "cifar100":
        dataset = datasets.CIFAR100
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )

    # Train transforms
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), 4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    traindata = dataset(
        root=root,
        train=True,
        transform=transform,
        download=True,
    )

    # split train dataset
    traindata_split = data.random_split(
        traindata, [int(traindata.data.shape[0] / n_nodes) for _ in range(n_nodes)])
    train_loader = [data.DataLoader(
        x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    test_loader = get_cifar_test(config, root)

    return train_loader, test_loader