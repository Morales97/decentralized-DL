
import os
import numpy as np
import pdb
from torchvision import datasets, transforms
import torch
import torch.utils.data as data
import random


def get_cifar_test(args, root, batch_size=100, test_transforms=None):
    '''
    Get CIFAR test set. 
    If val > 0, split into validation and test
    '''

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

    if test_transforms == 'no_norm':
        transform = transforms.Compose([transforms.ToTensor()])
    elif test_transforms:
        # for robustness experiments
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop((32, 32), 4),
            transforms.RandAugment(),
            transforms.ToTensor(), 
            normalize])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])



    dataset = dataset_loader(
        root=root,
        train=False,
        transform=transform,
        download=True,
    )

    test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)     
    return test_loader


def get_cifar_val_test(args, root, val=0, batch_size=100):
    '''
    DEPRECATED
    Get CIFAR test set. 
    If val > 0, split into validation and test
    '''

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

    if val > 0:
        n_samples = int(np.floor(len(dataset) * val))
        print(f'Splitting test set into val/test, with {n_samples}/{len(dataset)-n_samples} samples respectively')
        data_split = data.random_split(dataset, [n_samples, len(dataset)-n_samples], generator=torch.Generator().manual_seed(42))     # NOTE manual seed fixed for reproducible results
        val_loader = data.DataLoader(data_split[0], batch_size=batch_size, shuffle=False)     
        test_loader = data.DataLoader(data_split[1], batch_size=batch_size, shuffle=False)     
        return val_loader, test_loader
    else:
        test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)     
        return None, test_loader


def get_cifar(args, root, batch_size, val_fraction, iid=False, fraction=-1, noisy=False):
    '''
    Return CIFAR-10 or CIFAR-100 data loaders
    Optionally, return a subset (if fraction in [0,1])
    '''
    assert not (args.n_nodes[0] > 1 and val_fraction > 0), 'validation set for decentralized case not implemented'

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

    test_loader = get_cifar_test(args, root)

    # Optionally add label noise (CIFAR-100N from http://noisylabels.com/, 40% noisy labels)
    if noisy:
        if args.dataset == 'cifar10':
            noise_label = torch.load(os.path.join(root, 'cifar-100-python/CIFAR-10_human.pt'))
            assert noisy in ['aggre_label', 'worse_label', 'random_label1', 'random_label2', 'random_label3']
            noisy_label = noise_label[noisy] 
        elif args.dataset == 'cifar100':
            if noisy == '40':
                noise_label = torch.load(os.path.join(root, 'cifar-100-python/CIFAR-100_human.pt'))
                noisy_label = noise_label['noisy_label'] 
            else:
                # NOTE denote noise as 'sym_xx')
                noise_file = os.path.join(root, f'cifar-100-python/noise_{noisy}.pt')
                if os.path.exists(noise_file):
                    noisy_label = torch.load(noise_file)
                else:
                    # generate noise labels
                    noise_label = []
                    noise_percentage = int(noisy[-2:])  
                    idx = list(range(50000))
                    random.shuffle(idx)
                    num_noise = int(noise_percentage*50000)            
                    noise_idx = idx[:num_noise]
                    for i in range(50000):
                        if i in noise_idx:
                            # symmetric noise
                            if args.dataset=='cifar10':     # NOTE need to fix implementation to have this out of the 'cifar100' case
                                noiselabel = random.randint(0,9)
                            elif args.dataset=='cifar100':    
                                noiselabel = random.randint(0,99)
                            noise_label.append(noiselabel)                   
                        else:    
                            noise_label.append(traindata.targets[i]) 
                    torch.save(noise_label, noise_file)  
                    noisy_label = noise_label
                pdb.set_trace()

        traindata.targets = noisy_label.tolist()

    # Use a random subset of CIFAR
    if fraction > 0:
        assert fraction < 1
        n_samples = int(np.floor(len(traindata) * fraction))
        traindata = data.random_split(traindata, [n_samples, len(traindata)-n_samples], generator=torch.Generator().manual_seed(42))     # reduce size of traindata

    if args.n_nodes[0] == 1:
        if val_fraction > 0:
            n_samples = int(np.floor(len(traindata) * (1-val_fraction)))
            print(f'Splitting training set into train/val, with {n_samples}/{len(traindata)-n_samples} samples respectively')
            data_split = data.random_split(traindata, [n_samples, len(traindata)-n_samples], generator=torch.Generator().manual_seed(42))     # NOTE manual seed fixed for reproducible results
            train_loader = data.DataLoader(data_split[0], batch_size=batch_size, shuffle=True)     
            val_loader = data.DataLoader(data_split[1], batch_size=batch_size, shuffle=False)    
        else:
            train_loader = data.DataLoader(traindata, batch_size=batch_size, shuffle=True)     
            val_loader = test_loader

    else:   # decentralized. Note that train_loaders come in a list
        val_loader = test_loader
        if iid:
            sampler = data.RandomSampler(traindata, replacement=True, num_samples=batch_size)
            train_loader = data.DataLoader(traindata, sampler=sampler, batch_size=batch_size)
        else:
            # split train dataset
            traindata_split = data.random_split(traindata, [int(traindata.data.shape[0] / args.n_nodes[0]) for _ in range(args.n_nodes[0])])
            train_loader = [data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    return train_loader, val_loader, test_loader



def get_cifar_filtered_samples(args, root, teacher_model, samples_selected=None):
    '''
    For noisy CIFAR-100, filter the training set such that only samples predicted correctly by the teacher
    (i.e., supposedly without label noise) will be used

    samples_selected: If specified, it is a vector of bools indicating which samples to use in the subset
    '''

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    noise_label = torch.load(os.path.join(root, 'cifar-100-python/CIFAR-100_human.pt'))
    clean_label = noise_label['clean_label'] 
    noisy_label = noise_label['noisy_label'] 
    clean_labels = (clean_label == noisy_label)

    if samples_selected is not None:
        assert len(samples_selected) == 50000
        correct_wrt_noisy_labels = samples_selected
        noisy_correct_wrt_noise = np.array(correct_wrt_noisy_labels)[clean_labels==False].sum()
        clean_correct = np.array(correct_wrt_noisy_labels).sum() - noisy_correct_wrt_noise
    else:
        train_loader_no_shuffle = data.DataLoader(traindata, batch_size=100, shuffle=False)

        # predict with teacher
        correct = []
        correct_wrt_noisy_labels = []
        for i, (input, target) in enumerate(train_loader_no_shuffle):
            input = input.to(device)
            target = target.to(device)

            output = teacher_model(input)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).view(-1).tolist()
            noisy_target = torch.from_numpy(noisy_label[i*100:(i+1)*100]).to(device)
            correct_wrt_noisy_labels += pred.eq(noisy_target.view_as(pred)).view(-1).tolist()

        # compute accuracy on clean/noisy labels
        n_clean = clean_labels.sum()
        n_noisy = 50000 - n_clean
        clean_correct = np.array(correct)[clean_labels].sum()
        noisy_correct = np.array(correct)[clean_labels==False].sum()
        noisy_correct_wrt_noise = np.array(correct_wrt_noisy_labels)[clean_labels==False].sum()
        acc_on_clean = clean_correct / n_clean
        acc_on_noisy = noisy_correct / n_noisy
        noisy_labels_fitted = noisy_correct_wrt_noise / n_noisy

        print('\n ~~ ON CLEAN LABELS ~~')
        print(f'Teacher model train accuracy (wrt clean labels): {np.sum(correct)/len(traindata.targets)*100:.2f}%')
        print(f'Teacher model train accuracy on {n_clean} clean-label samples: {acc_on_clean*100:.2f}%')
        print(f'Teacher model train accuracy on {n_noisy} noisy-label samples: {acc_on_noisy*100:.2f}%')
        
        print('\n ~~ ON NOISY LABELS ~~')
        print(f'\nTeacher model train accuracy (wrt noisy labels): {np.sum(correct_wrt_noisy_labels)/len(traindata.targets)*100:.2f}%')
        print(f'Noise fitted: Noisy-label samples (out of {n_noisy}) where teacher predicted the noisy label: {noisy_labels_fitted*100:.2f}%')
        
        np.save(root + '/filtered_dataset_40.npy', np.array(correct_wrt_noisy_labels))

    # use noisy labels dataset, filtering out samples which where not predicted correctly by teacher (suspicious of noise!)
    traindata.targets = noisy_label.tolist()
    filtered_dataset = data.Subset(traindata, np.where(correct_wrt_noisy_labels)[0])
    train_loader = [data.DataLoader(filtered_dataset, batch_size=args.batch_size[0], shuffle=True)]
    
    print('\n ~~ FILTERED DATASET ~~')
    print(f'The new dataset contains {len(train_loader[0].dataset)} samples. {clean_correct} clean-labeled samples + {noisy_correct_wrt_noise} noisy-label.')

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