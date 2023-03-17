
import os
import numpy as np
import pdb
from torchvision import datasets, transforms, io
import torch
import torch.utils.data as data
import glob
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def get_animal_test(args, root, batch_size=100, test_transforms=None):
    '''
    Get CIFAR test set. 
    If val > 0, split into validation and test
    '''

    transform = transforms.Compose([transforms.ToTensor()
                                    #, normalize
                                    ])

    dataset = CustomDataset(root=root+'/animal10/testing/*', transform=transform)
    test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)     
    return test_loader

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = glob.glob(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = Image.open(self.files[idx])
        if self.transform:
            data = self.transform(data)
        label = int(self.files[idx].split('/')[-1][0]) # label is the first char of the file, 0-9
        return data, label 



def get_animal(args, root, batch_size, val_fraction):

    # Train transforms
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), 4),
            transforms.ToTensor(),
            #normalize,
        ]
    )
    
    dataset = CustomDataset(root=root+'/animal10/training/*', transform=transform)
    train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)     

    test_loader = get_animal_test(args, root)
    val_loader = test_loader

    return train_loader, val_loader, test_loader
