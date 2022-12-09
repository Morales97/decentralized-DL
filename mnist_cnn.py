import numpy as np
import pdb
from data_helpers import get_mnist, get_heterogeneous_mnist
from topology import get_diff_matrix, diffuse
# from plot_helpers import plot_calibration_histogram, plot_heatmaps, plot_node_disagreement
import time 
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from models import get_model
import torch.nn.functional as F
from utils import save_experiment

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

    return loss.item()


def compute_node_disagreement(config, models, n_nodes):
    avg_model = get_model(config, 'cpu')
    state_dict_avg = avg_model.state_dict()
    models_sd = [models[i].state_dict() for i in range(n_nodes)]
    L2_diff = 0
    for key in state_dict_avg.keys():
        state_dict_avg[key] = torch.stack(
                                        [models_sd[i][key] for i in range(n_nodes)], dim=0
                                        ).sum(0) / n_nodes
        L2_diff += torch.stack(
                            [(state_dict_avg[key] - models_sd[i][key])**2 for i in range(n_nodes)], dim=0
                            ).sum() / n_nodes
    
    return L2_diff

def train_mnist(config, expt):

    decentralized = (expt['topology'] != 'centralized') 

    # central training
    if not decentralized:
        n_nodes = 1
        batch_size = config['batch_size'] * config['n_nodes']
        if 'data_split' not in config.keys():
            config['data_split'] = 'yes'   # 'yes' will train epoch by epoch. 'no' will sample WITH REPLACEMENT
    
    # decentralized
    else:
        n_nodes = config['n_nodes']
        batch_size = config['batch_size']
        if 'data_split' not in config.keys():
            config['data_split'] = 'yes'   # default: split dataset between workers

    # data
    train_loader, test_loader = get_mnist(config, batch_size)
    if config['data_split'] == 'yes':
        train_loader_lengths = [len(t) for t in train_loader]
        train_loader_iter = [iter(t) for t in train_loader]

    # init
    torch.manual_seed(0)    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = [get_model(config, device) for _ in range(n_nodes)]
    opts = [optim.SGD(model.parameters(), lr=config['lr']) for model in models]
    if config['same_init']:
        for i in range(1, len(models)):
            models[i].load_state_dict(models[0].state_dict())

    comm_matrix = get_diff_matrix(expt, n_nodes)
    

    accuracies = []
    test_losses = []
    train_losses = []
    node_disagreement = []

    ts_total = time.time()
    ts = time.time()
    for step in range(config['steps']):
        
        if config['data_split'] == 'yes':
            for i, l in enumerate(train_loader_lengths):
                if step%l == 0:
                    train_loader_iter[i] = iter(train_loader[i])

        # local update
        train_loss = 0
        for i in range(len(models)):
            if config['data_split'] == 'yes':
                train_loss += worker_local_step(models[i], opts[i], train_loader_iter[i], device)
            elif config['data_split'] == 'no':
                train_loss += worker_local_step(models[i], opts[i], iter(train_loader), device)
        train_losses.append(train_loss/n_nodes)
        
        if 'train_loss_th' in config.keys() and train_losses[-1] < config['train_loss_th']:
            return accuracies, test_losses, train_losses, None

        # gossip
        diffuse(comm_matrix, models, step, expt)

        if decentralized:
            L2_diff = compute_node_disagreement(config, models, n_nodes)
            node_disagreement.append(L2_diff)


        # evaluate
        if (step+1) % config['steps_eval'] == 0:
            acc_workers = []
            loss_workers = []
            ts_eval = time.time()
            for model in models:
                test_loss, acc = evaluate_model(model, test_loader, device)
                acc_workers.append(acc)
                loss_workers.append(test_loss)
            accuracies.append(float(np.array(acc_workers).mean()*100))
            test_losses.append(np.array(loss_workers).mean())
            print('Step % d -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' % (step, accuracies[-1], test_losses[-1], train_losses[-1], time.time() - ts_total, time.time() - ts, time.time() - ts_eval))     
            ts = time.time()
            # print('time evaluating: %.2f' % (time.time() - ts_eval))
            
            # stop if threhsold reached
            if 'acc_th' in config.keys() and config['acc_th'] < acc:
                return accuracies, test_losses, train_losses, None
            if 'loss_th' in config.keys() and test_loss < config['loss_th']:
                return accuracies, test_losses, train_losses, None

    return accuracies, test_losses, train_losses, node_disagreement

config = {
    'n_nodes': 15,
    'batch_size': 20,
    'lr': 1,
    'steps': 100,
    'steps_eval': 50,
    'data_split': 'yes',     # NOTE 'no' will sample with replacement from the FULL dataset, which will be truly IID
    'same_init': True,
    'small_test_set': True,
    'net': 'mlp', # 'convnet'
    # 'net': 'convnet',
}

expt = {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0}
# expt = {'topology': 'solo', 'local_steps': 0}
# expt = {'topology': 'fully_connected', 'local_steps': 0}
# expt = {'topology': 'fully_connected', 'local_steps': 58}
# expt = {'topology': 'random', 'degree': 7, 'label': 'Fully connected', 'local_steps': 0}
# expt = {'topology': 'exponential_graph', 'local_steps': 0}


if __name__ == '__main__':
    # train_mnist_centralized(config)
    acc, test_loss, train_loss, consensus = train_mnist(config, expt)
    save_experiment({**config, **expt}, acc, test_loss, train_loss, consensus, filename='experiments_mnist/results/test')
    # model = MLP()
    # model = ConvNet()
    # summary(model, (1, 28, 28))
    # get_heterogeneous_mnist(10, 20)