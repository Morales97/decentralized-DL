
import os
import numpy as np
import pdb
from torchvision import datasets, transforms
import torch
import torch.utils.data as data


def get_mnist_test(root, batch_size):

    testdata = datasets.MNIST(root, train=False, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_loader = data.DataLoader(
        testdata,
        batch_size=batch_size,
        shuffle=True)

    return test_loader


def get_mnist_iid(root, batch_size):
    '''
    Return the full dataset, random sampling with replacement
    '''
    traindata = datasets.MNIST(root, train=True, download=True,
                               transform=transforms.Compose(
                                   [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                               )
    
    sampler = data.RandomSampler(
        traindata, replacement=True, num_samples=batch_size)
    train_loader = data.DataLoader(
        traindata, sampler=sampler, batch_size=batch_size)

    test_loader = get_mnist_test(root, 32)

    return train_loader, test_loader


def get_mnist_split(root, n_nodes, batch_size):
    '''
    Split dataset randomly between workers -> breaks IID
    '''
    traindata = datasets.MNIST(root, train=True, download=True,
                               transform=transforms.Compose(
                                   [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                               )

    traindata_split = data.random_split(
        traindata, [int(traindata.data.shape[0] / n_nodes) for _ in range(n_nodes)])
    train_loader = [data.DataLoader(
        x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    test_loader = get_mnist_test(root, 32)

    return train_loader, test_loader


def get_heterogeneous_mnist(root, n_nodes, batch_size, p):
    '''
    Return Heterogeneous MNIST dataset as:
        Fraction (p) of data with non-IID split. Label skew, assign samples from two random classes per node.
        Fraction (1-p) of data with IID split.
    '''

    num_nonidd_samples = int(60000 * p)
    num_iid_samples = 60000 - num_nonidd_samples

    traindata = datasets.MNIST(root, train=True, download=True,
                               transform=transforms.Compose(
                                   [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                               )

    target_labels = torch.stack([traindata.targets == i for i in range(10)])
    target_labels_noniid = target_labels[:, :num_nonidd_samples]
    # target_labels_iid = target_labels[:, num_nonidd_samples:]

    # NON-IID
    target_labels_sorted = torch.cat(
        [torch.where(target_labels_noniid[i:i+1].sum(0))[0] for i in range(n_nodes)])
    n_shards = n_nodes * 2  # assign 2 random shards to each node
    n_samples_per_shard = num_nonidd_samples // n_shards
    random_shard_perm = np.random.permutation(np.arange(n_shards))

    indices_per_node = []
    for i in range(n_nodes):
        shard_1 = random_shard_perm[2*i]
        shard_2 = random_shard_perm[2*i+1]
        samples_shard_1 = target_labels_sorted[shard_1 *
                                               n_samples_per_shard: (shard_1+1)*n_samples_per_shard]
        samples_shard_2 = target_labels_sorted[shard_2 *
                                               n_samples_per_shard: (shard_2+1)*n_samples_per_shard]
        indices_per_node.append(np.concatenate(
            (samples_shard_1, samples_shard_2)))

    traindata_split_noniid = [data.Subset(
        traindata, tl) for tl in indices_per_node]

    # IID
    indices_iid_per_node = []
    num_iid_samples_per_node = num_iid_samples // n_nodes
    for i in range(n_nodes):
        start = num_nonidd_samples + i*num_iid_samples_per_node
        end = num_nonidd_samples + (i+1)*num_iid_samples_per_node
        indices_iid_per_node.append(np.arange(start, end))

    traindata_split_iid = [data.Subset(traindata, tl)
                           for tl in indices_iid_per_node]

    # Combine IID and non-IID splits
    train_loader = [data.DataLoader(data.ConcatDataset([x, z]), batch_size=batch_size, shuffle=True)
                    for x, z in zip(traindata_split_iid, traindata_split_noniid)]

    # Print for check
    counts = np.zeros(10)
    for x, z in zip(indices_per_node, indices_iid_per_node):
        label_distr = np.unique(
            traindata.targets[np.concatenate([x, z])].numpy(), return_counts=True)
        samples_node = np.zeros(10)
        samples_node[label_distr[0]] += label_distr[1]
        counts += samples_node
        print('Samples per class (at each node): ' + str(samples_node))
    print('Total samples used per class: ' + str(counts))

    test_loader = get_mnist_test(root, 32)

    return train_loader, test_loader
