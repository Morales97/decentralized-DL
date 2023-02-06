
import os
import numpy as np
import pdb
from torchvision import datasets, transforms
import torch
import torch.utils.data as data


def get_cifar_test(args, root):

    # decide normalize parameter.
    if args.dataset == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    elif args.dataset == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )

    transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    dataset = dataset_loader(
        root=root,
        train=False,
        transform=transform,
        download=True,
    )

    return data.DataLoader(dataset, batch_size=100, shuffle=False)


def get_cifar(args, root, batch_size, iid=True, fraction=-1):
    '''
    Return CIFAR-10 or CIFAR-100 data loaders
    Optionally, return a subset (if fraction in [0,1])
    '''

    # decide normalize parameter.
    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    elif args.dataset == "cifar100":
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

    # Use a random subset of CIFAR
    if fraction > 0:
        assert fraction < 1
        n_samples = int(np.floor(len(traindata) * fraction))
        traindata_split = data.random_split(traindata, [n_samples, len(traindata)-n_samples], generator=torch.Generator().manual_seed(42))     # manual seed fixed for reproducible results
        sampler = data.RandomSampler(traindata_split[0], replacement=True, num_samples=batch_size)  # to sample with replacement (IID)
        train_loader = data.DataLoader(traindata_split[0], sampler=sampler, batch_size=batch_size)
    
    # Use full CIFAR
    else:
        if iid:
            sampler = data.RandomSampler(
                traindata, replacement=True, num_samples=batch_size)
            train_loader = data.DataLoader(
                traindata, sampler=sampler, batch_size=batch_size)
        else:
            # split train dataset
            traindata_split = data.random_split(
                traindata, [int(traindata.data.shape[0] / args.n_nodes[0]) for _ in range(args.n_nodes[0])])
            train_loader = [data.DataLoader(
                x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    test_loader = get_cifar_test(args, root)

    return train_loader, test_loader


# def create_ffcv_dataset():
#     datasets = {
#         'train': datasets.CIFAR10('/tmp', train=True, download=True),
#         'test': datasets.CIFAR10('/tmp', train=False, download=True)
#     }

#     for (name, ds) in datasets.items():
#         writer = DatasetWriter(f'/tmp/cifar_{name}.beton', {
#             'image': RGBImageField(),
#             'label': IntField()
#         })
#         writer.from_indexed_dataset(ds)