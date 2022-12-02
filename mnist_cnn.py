import numpy as np
import pdb
import matplotlib.pyplot as plt
from data_helpers import get_mnist
from topology import get_diff_matrix, diffuse
# from plot_helpers import plot_calibration_histogram, plot_heatmaps, plot_node_disagreement
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

def evaluate_model(model, data_loader, device):
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc

def worker_local_step(model, opt, train_loader_iter, device):
    input, target = next(train_loader_iter)
    input = input.to(device)
    target = target.to(device)

    model.train()
    output = model(input)
    opt.zero_grad()
    loss = F.nll_loss(output, target)
    loss.backward()
    opt.step()


def train_mnist_iid(config, expt):
    '''
    Train decentralized MNIST on IID data. 
    '''

    decentralized = (expt['topology'] != 'centralized') 

    # central training
    if not decentralized:
        n_nodes = 1
        config['batch_size'] = config['batch_size'] * config['n_nodes']
        config['data_split'] = 'no'
    
    # decentralized
    else:
        n_nodes = config['n_nodes']
        if 'data_split' not in config.keys():
            config['data_split'] = 'yes'   # default: split dataset between workers

    # data
    train_loader, test_loader = get_mnist(config)
    if config['data_split'] == 'yes':
        train_loader_lengths = [len(t) for t in train_loader]
        train_loader_iter = [iter(t) for t in train_loader]

    # init
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = [Net().to(device) for _ in range(n_nodes)]
    opts = [optim.SGD(model.parameters(), lr=0.1) for model in models]
    comm_matrix = get_diff_matrix(expt, n_nodes)

    accuracies = []
    test_losses = []

    ts = time.time()
    for step in range(config['steps']):
        
        if config['data_split'] == 'yes':
            for i, l in enumerate(train_loader_lengths):
                if step%l == 0:
                    train_loader_iter[i] = iter(train_loader[i])

        # local update
        for i in range(len(models)):
            if config['data_split'] == 'yes':
                worker_local_step(models[i], opts[i], train_loader_iter[i], device)
            elif config['data_split'] == 'no':
                worker_local_step(models[i], opts[i], iter(train_loader), device)

        # gossip
        diffuse(comm_matrix, models, step, expt)

        # evaluate
        if (step+1) % config['steps_eval'] == 0:
            acc_workers = []
            loss_workers = []
            ts_eval = time.time()
            for model in models:
                test_loss, acc = evaluate_model(model, test_loader, device)
                acc_workers.append(acc)
                loss_workers.append(test_loader)
            accuracies.append(float(np.mean(acc)*100))
            test_losses.append(np.mean(test_loss))
            print('Step % d -- Test accuracy: %.2f -- Test loss: %.3f -- Total time: %.2f s' % (step, accuracies[-1], test_losses[-1], time.time() - ts))     
            print('time evaluating: %.2f' % (time.time() - ts_eval))
            
config = {
    'n_nodes': 4,
    'batch_size': 16,
    'steps': 1000,
    'steps_eval': 50,
    'data_split': 'no'
}

# expt = {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0}
# expt = {'topology': 'solo', 'local_steps': 0}
expt = {'topology': 'fully_connected', 'local_steps': 0}
# expt = {'topology': 'fully_connected', 'local_steps': 58}
# expt = {'topology': 'random', 'degree': 7, 'label': 'Fully connected', 'local_steps': 0}
# expt = {'topology': 'exponential_graph', 'local_steps': 0}


if __name__ == '__main__':
    # train_mnist_centralized(config)
    train_mnist_iid(config, expt)
    # model = Net2()
    # summary(model, (1, 28, 28))