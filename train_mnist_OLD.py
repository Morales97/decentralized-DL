from turtle import st
import numpy as np
import pdb
from data.data import get_data
from topology import get_diff_matrix, diffuse, get_average_model
# from plot_helpers import plot_calibration_histogram, plot_heatmaps, plot_node_disagreement
import time 
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from model.model import get_model
import torch.nn.functional as F
from helpers.utils import save_experiment
from helpers.gradient_var import *
from helpers.consensus import *

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



########################################################################################


def train_mnist(config, expt):

    decentralized = (expt['topology'] != 'centralized') 
    if config['p_label_skew'] > 0:
        assert config['data_split'] == 'yes', 'Sampling with replacement only available if split are not heterogeneous'
    if 'eval_on_average_model' not in config.keys():
        config['eval_on_average_model'] = False
    if 'steps_weight_distance' not in config.keys():
        config['steps_weight_distance'] = -1
    if 'warmup_steps' not in config.keys():
            config['warmup_steps'] = 0

    # central training
    if not decentralized:
        n_nodes = 1
        batch_size = config['batch_size'] * config['n_nodes']    
    # decentralized
    else:
        n_nodes = config['n_nodes']
        batch_size = config['batch_size']

    # data
    if 'data_split' not in config.keys():
        config['data_split'] = 'yes'   # default: split dataset between workers
    train_loader, test_loader = get_data(config, n_nodes, batch_size)
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
    if config['steps_weight_distance'] > 0:
        init_model = get_model(config, device)
        init_model.load_state_dict(models[0].state_dict())

    comm_matrix = get_diff_matrix(expt, n_nodes)
    # print(comm_matrix)

    accuracies = []
    test_losses = []
    train_losses = []
    node_disagreement = []
    gradient_var = []
    gradient_coh = []
    weight_distance = []

    ts_total = time.time()
    ts = time.time()
    for step in range(config['steps']):
        
        if config['data_split'] == 'yes':
            for i, l in enumerate(train_loader_lengths):
                if step%l == 0:
                    train_loader_iter[i] = iter(train_loader[i])

        if step < config['warmup_steps']:
            lr = config['lr'] * (step+1) / config['warmup_steps']
            for opt in opts:
                for g in opt.param_groups:
                    g['lr'] = lr

        # local update
        train_loss = 0
        for i in range(len(models)):
            if config['data_split'] == 'yes':
                train_loss += worker_local_step(models[i], opts[i], train_loader_iter[i], device)
            elif config['data_split'] == 'no':
                train_loss += worker_local_step(models[i], opts[i], iter(train_loader), device)
        train_losses.append(train_loss/n_nodes) 
        
        # if 'train_loss_th' in config.keys() and train_losses[-1] < config['train_loss_th']:
        #     return accuracies, test_losses, train_losses, None

        # gossip
        diffuse(comm_matrix, models, step, expt)

        if decentralized:
            # L2_diff = compute_node_disagreement(config, models, n_nodes)
            # node_disagreement.append(L2_diff)
            pass

        if 'steps_grad_var' in config.keys() and (step+1) % config['steps_grad_var'] == 0:

            if config['data_split'] == 'yes':
                grad_var, grad_coh = compute_var_and_coh(config, models, opts, train_loader_iter, train_loader, test_loader, device)
            elif config['data_split'] == 'no':
                grad_var, grad_coh = compute_var_and_coh_iid(config, models, opts, train_loader, device)
            gradient_var.append(grad_var)
            gradient_coh.append(grad_coh)

        # evaluate
        if not config['eval_on_average_model']:
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
                # if 'acc_th' in config.keys() and config['acc_th'] < acc:
                #     return accuracies, test_losses, train_losses, None
                # if 'loss_th' in config.keys() and test_loss < config['loss_th']:
                #     return accuracies, test_losses, train_losses, None

        # evaluate on averaged model
        else:
            if (step+1) % config['steps_eval'] == 0:
                ts_eval = time.time()
                model = get_average_model(config, device, models)
                test_loss, acc = evaluate_model(model, test_loader, device)
                accuracies.append(float(acc*100))
                test_losses.append(test_loss)
                print('Step % d -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' % (step, float(acc*100), test_loss, np.mean(train_losses[-25:]), time.time() - ts_total, time.time() - ts, time.time() - ts_eval))     
                ts = time.time()

        if (step+1) % config['steps_weight_distance'] == 0 and config['steps_weight_distance'] > 0:
            model = get_average_model(config, device, models)
            dist = compute_weight_distance(config, model, init_model)
            weight_distance.append(dist)
            # pdb.set_trace()
            # print('Step %d -- Weight dist: %.3f' % (step, dist))

            



    return accuracies, test_losses, train_losses, node_disagreement, gradient_var, weight_distance


config = {
    'n_nodes': 15,
    'batch_size': 20,
    'lr': 0.1,
    'steps': 1000,
    'steps_eval': 100,
    # 'steps_grad_var': 1,
    'data_split': 'yes',     # NOTE 'no' will sample with replacement from the FULL dataset, which will be truly IID
    'same_init': True,
    'p_label_skew': 0,
    # 'net': 'mlp', # 'convnet'
    'net': 'convnet',
    'steps_weight_distance': 5,
}

# expt = {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0}
# expt = {'topology': 'solo', 'local_steps': 0}
# expt = {'topology': 'fully_connected', 'local_steps': 0}
# expt = {'topology': 'fully_connected', 'local_steps': 50}
# expt = {'topology': 'random', 'degree': 4, 'local_steps': 0}
# expt = {'topology': 'exponential_graph', 'local_steps': 0}
expt = {'topology': 'ring', 'local_steps': 0}

if __name__ == '__main__':
    acc, test_loss, train_loss, consensus = train_mnist(config, expt)
