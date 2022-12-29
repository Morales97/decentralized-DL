from turtle import st
import numpy as np
import pdb
from data_helpers import get_mnist, get_heterogeneous_mnist
from topology import get_diff_matrix, diffuse, get_average_model
# from plot_helpers import plot_calibration_histogram, plot_heatmaps, plot_node_disagreement
import time 
import torch
import torch.nn as nn
import torch.optim as optim
from models import get_model
import torch.nn.functional as F
from utils import save_experiment, get_expt_name
from helpers.gradient_var import *
from helpers.consensus import *

import wandb


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

class Logger:
    def __init__(self, wandb):
        self.wandb = wandb
        self.accuracies = []
        self.accuracies_per_node = []
        self.test_losses = []
        self.test_losses_per_node = []
        self.train_losses = []
        self.weight_distance = []

    def log_step(self, step, epoch, train_loss, ts_total, ts_step):
        self.train_losses.append(train_loss)
        log = {
            'Train Loss': train_loss,
            'Iteration': step,
            'Epoch': epoch,
            'Total time': time.time() - ts_total,
            'Time/step': time.time() - ts_step,
            }
        self.wandb.log(log)

    def log_eval(self, step, epoch, acc, test_loss, ts_eval, ts_steps_eval):
        self.accuracies.append(acc)
        self.test_losses.append(test_loss)
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Test Accuracy': acc,
            'Test Loss': test_loss,
            'Time/eval': time.time() - ts_eval,
            'Time since last eval': time.time() - ts_steps_eval
            }
        self.wandb.log(log)

    def log_eval_random_node(self, step, epoch, acc, test_loss):
        self.accuracies.append(acc)
        self.test_losses.append(test_loss)
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Test Accuracy [random node]': acc,
            'Test Loss [random node]': test_loss,
            }
        self.wandb.log(log)

    def log_eval_per_node(self, step, epoch, acc, test_loss, acc_nodes, loss_nodes, acc_avg, loss_avg, ts_eval, ts_steps_eval):
        self.accuracies.append(acc)
        self.test_losses.append(test_loss)
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Test Accuracy': acc,
            'Test Loss': test_loss,
            'Test Accuracy [min]': min(acc_nodes),
            'Test Accuracy [max]': max(acc_nodes),
            'Test Loss [min]': min(loss_nodes),
            'Test Loss [max]': max(loss_nodes),
            'Test Accuracy [avg model]': acc_avg, 
            'Test Loss [avg model]': loss_avg,
            'Time/eval': time.time() - ts_eval,
            'Time since last eval': time.time() - ts_steps_eval
            }
        self.wandb.log(log)

    def log_weight_distance(self, step, epoch, weight_dist):
        self.weight_distance.append(weight_dist)
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Weight distance to init': weight_dist,
        }
        self.wandb.log(log)

########################################################################################


def train_mnist(config, expt, wandb):

    decentralized = (expt['topology'] != 'centralized') 
    if config['p_label_skew'] > 0:
        assert config['data_split'] == 'yes', 'Sampling with replacement only available if split are not heterogeneous'
    if 'eval_on_average_model' not in config.keys():
        config['eval_on_average_model'] = False

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
    train_loader, test_loader = get_mnist(config, n_nodes, batch_size)
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

    logger = Logger(wandb)

    ts_total = time.time()
    ts_steps_eval = time.time()
    epoch = 0
    for step in range(config['steps']):
        
        if config['data_split'] == 'yes':
            for i, l in enumerate(train_loader_lengths):
                if step%l == 0:
                    train_loader_iter[i] = iter(train_loader[i])

        # local update
        train_loss = 0
        ts_step = time.time()
        for i in range(len(models)):
            if config['data_split'] == 'yes':
                train_loss += worker_local_step(models[i], opts[i], train_loader_iter[i], device)
            elif config['data_split'] == 'no':
                train_loss += worker_local_step(models[i], opts[i], iter(train_loader), device)
        epoch += n_nodes*batch_size / 60000
        train_loss /= n_nodes
        logger.log_step(step, epoch, train_loss, ts_total, ts_step)

        # gossip
        diffuse(comm_matrix, models, step, expt)

        # evaluate per node
        if not config['eval_on_average_model'] or not decentralized:
            if (step+1) % config['steps_eval'] == 0 or (step+1) == config['steps']:
                acc_workers = []
                loss_workers = []
                ts_eval = time.time()
                for model in models:
                    test_loss, acc = evaluate_model(model, test_loader, device)
                    acc_workers.append(acc)
                    loss_workers.append(test_loss)
                acc = float(np.array(acc_workers).mean()*100)
                test_loss = np.array(loss_workers).mean()

                model = get_average_model(config, device, models)
                test_loss_avg, acc_avg = evaluate_model(model, test_loader, device)
                logger.log_eval_per_node(step, epoch, acc, test_loss, acc_workers, loss_workers, acc_avg, test_loss_avg, ts_eval, ts_steps_eval)
                # print('Step % d -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' % (step, acc, test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))     
                print('Epoch %.3f -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' % (epoch, acc, test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))     
                ts_steps_eval = time.time()

        # evaluate on averaged model
        else:
            if (step+1) % config['steps_eval'] == 0 or (step+1) == config['steps']:
                ts_eval = time.time()
                model = get_average_model(config, device, models)
                test_loss, acc = evaluate_model(model, test_loader, device)
                logger.log_eval(step, epoch, float(acc*100), test_loss, ts_eval, ts_steps_eval)
                print('Step % d -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' % (step, float(acc*100), test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))     
                
                if not decentralized:   
                    # evaluate also on a random worker
                    test_loss, acc = evaluate_model(model[0], test_loader, device)
                    logger.log_eval_random_node(step, epoch, float(acc*100), test_loss)
                
                ts_steps_eval = time.time()

        # weight distance to init
        if (step+1) % config['steps_weight_distance'] == 0 and config['steps_weight_distance'] > 0:
            model = get_average_model(config, device, models)
            dist = compute_weight_distance(config, model, init_model)
            logger.log_weight_distance(step, epoch, dist)


    return logger.accuracies, logger.test_losses, logger.train_losses, None, None, logger.weight_distance


config = {
    'n_nodes': 15,
    'batch_size': 32,
    'lr': 0.3,
    'steps': 1000,
    'steps_eval': 100,
    'data_split': 'yes',     # NOTE 'no' will sample with replacement from the FULL dataset, which will be truly IID
    'same_init': True,
    'small_test_set': False,
    'p_label_skew': 0,
    'net': 'convnet',
    'wandb': True,
    'steps_weight_distance': 25,
    'eval_on_average_model': False,
}


expt2 = {'topology': 'centralized', 'label': 'Centralized', 'local_steps': 0}
expt3 = {'topology': 'centralized', 'label': 'Centralized, LR warm up (100)', 'local_steps': 0, 'warmup_steps': 100}

#expt = {'topology': 'solo', 'local_steps': 0}
# expt = {'topology': 'centralized', 'label': 'Centralized', 'local_steps': 0}
# expt = {'topology': 'fully_connected', 'local_steps': 0}
# expt = {'topology': 'fully_connected', 'local_steps': 50}
# expt = {'topology': 'random', 'degree': 4, 'local_steps': 0}
# expt = {'topology': 'exponential_graph', 'local_steps': 0}
# expt = {'topology': 'ring', 'local_steps': 0}
expt = {'topology': 'ring', 'local_steps': 0, 'data_split': 'no', 'eval_on_average_model': True}

if __name__ == '__main__':

    for lr in [0.05, 0.1, 0.15, 0.3, 0.6, 0.9]:
        for bs in [16, 32, 64, 128, 256, 512]:
        # for bs in [256, 512]:
            # if lr == 0.05 and bs == 256:
            #     pass
            # else:
            steps = int(25 * 60000 // (bs*15))
            steps_eval = min(100, (steps // 10))
            config['steps'] = steps
            config['steps_eval'] = steps_eval
            config['lr'] = lr
            config['batch_size'] = bs

            for i in range(5):
                name = get_expt_name(config, expt)
                wandb.init(name=name, dir='.', config={**config, **expt}, reinit=True, project='MLO-MNIST-BSvsLR', entity='morales97')
                acc, test_loss, train_loss, _, _, _ = train_mnist(config, expt, wandb)
                wandb.finish()



    # for i in range(5):

    #     name = get_expt_name(config, expt)
    #     wandb.init(name=name, dir='.', config={**config, **expt}, reinit=True, project='MLO-MNIST', entity='morales97')
    #     acc, test_loss, train_loss, _, _, _ = train_mnist(config, expt, wandb)
    #     wandb.finish()

        # name = get_expt_name(config, expt2)
        # wandb.init(name=name, dir='.', config={**config, **expt2}, reinit=True, project='MLO-MNIST', entity='morales97')
        # acc, test_loss, train_loss, _, _, _ = train_mnist(config, expt2, wandb)
        # wandb.finish()

        # name = get_expt_name(config, expt3)
        # wandb.init(name=name, dir='.', config={**config, **expt3}, reinit=True, project='MLO-MNIST', entity='morales97')
        # acc, test_loss, train_loss, _, _, _ = train_mnist(config, expt3, wandb)
        # wandb.finish()