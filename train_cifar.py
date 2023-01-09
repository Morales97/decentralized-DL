import numpy as np
import pdb
from data.data import get_data
from topology import get_diff_matrix, diffuse, get_average_model
import time
import torch
import torch.optim as optim
from model.model import get_model
import torch.nn.functional as F
from helpers.utils import save_experiment, get_expt_name
from helpers.gradient_var import *
from helpers.consensus import *
from helpers.logger import Logger
import wandb


def evaluate_model(model, data_loader, device):
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # data = data.to(device)
            output = model(data)
            # output = model(data[None, ...])
            # sum up batch loss
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
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
    loss = F.cross_entropy(output, target)
    loss.backward()
    opt.step()

    return loss.item()


########################################################################################


def train_cifar(config, expt, wandb):

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
    train_loader, test_loader = get_data(config, n_nodes, batch_size)
    if config['data_split'] == 'yes':
        train_loader_lengths = [len(t) for t in train_loader]
        train_loader_iter = [iter(t) for t in train_loader]

    # init
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = [get_model(config, device) for _ in range(n_nodes)]
    opts = [optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, nesterov=True, weight_decay=1e-4) for model in models]
    if config['same_init']:
        for i in range(1, len(models)):
            models[i].load_state_dict(models[0].state_dict())

    comm_matrix = get_diff_matrix(expt, n_nodes)
    # print(comm_matrix)

    logger = Logger(wandb)

    ts_total = time.time()
    ts_steps_eval = time.time()
    epoch = 0
    for step in range(config['steps']):

        if config['data_split'] == 'yes':
            for i, l in enumerate(train_loader_lengths):
                if step % l == 0:
                    train_loader_iter[i] = iter(train_loader[i])

        if step < config['warmup_steps']:
            lr = config['lr'] * (step+1) / config['warmup_steps']
            for opt in opts:
                for g in opt.param_groups:
                    g['lr'] = lr

        # decay lr at 50% and 75%
        if step == config['steps']//2 or step == config['steps']//4*3:
            for opt in opts:
                for g in opt.param_groups:
                    g['lr'] = g['lr']/10
            print('lr decayed to %.4f' % g['lr'])

        # local update
        train_loss = 0
        ts_step = time.time()
        for i in range(len(models)):
            if config['data_split'] == 'yes':
                train_loss += worker_local_step(models[i],
                                                opts[i], train_loader_iter[i], device)
            elif config['data_split'] == 'no':
                train_loss += worker_local_step(models[i],
                                                opts[i], iter(train_loader), device)
        epoch += n_nodes*batch_size / 50000
        train_loss /= n_nodes
        logger.log_step(step, epoch, train_loss, ts_total, ts_step)

        # gossip
        diffuse(comm_matrix, models, step, expt)

        # evaluate per node
        if (not config['eval_on_average_model']) and decentralized:
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
                test_loss_avg, acc_avg = evaluate_model(
                    model, test_loader, device)
                logger.log_eval_per_node(step, epoch, acc, test_loss, acc_workers,
                                         loss_workers, acc_avg, test_loss_avg, ts_eval, ts_steps_eval)
                # print('Step % d -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' % (step, acc, test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))
                print('Epoch %.3f -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                      (epoch, acc, test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))
                ts_steps_eval = time.time()

        # evaluate on averaged model
        else:
            if (step+1) % config['steps_eval'] == 0 or (step+1) == config['steps']:
                ts_eval = time.time()
                model = get_average_model(config, device, models)
                test_loss, acc = evaluate_model(model, test_loader, device)
                logger.log_eval(step, epoch, float(acc*100),
                                test_loss, ts_eval, ts_steps_eval)
                print('Step % d -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                      (step, float(acc*100), test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))

                if decentralized:
                    # evaluate also on a random worker
                    test_loss, acc = evaluate_model(models[0], test_loader, device)
                    logger.log_eval_random_node(step, epoch, float(acc*100), test_loss)

                ts_steps_eval = time.time()


    return logger.accuracies, logger.test_losses, logger.train_losses, None, None, logger.weight_distance


config = {
    'n_nodes': 8,
    'batch_size': 128,
    'lr': 0.2*16,
    'steps': 50000//(128*16)*300,
    'warmup_steps': 50000//(128*16)*5,
    'steps_eval': 50000//(128*4),
    'data_split': 'yes', # NOTE 'no' will sample with replacement from the FULL dataset, which will be truly IID
    'same_init': True,
    'p_label_skew': 0,
    'net': 'resnet18',
    'wandb': True,
    'eval_on_average_model': False,
    'dataset': 'cifar10',
}


# expt = {'topology': 'centralized', 'label': 'Centralized', 'local_steps': 0}
# expt3 = {'topology': 'centralized', 'label': 'Centralized, LR warm up (100)', 'local_steps': 0, 'warmup_steps': 100}

# expt = {'topology': 'solo', 'local_steps': 0}
# expt = {'topology': 'centralized', 'label': 'Centralized', 'local_steps': 0}
# expt = {'topology': 'fully_connected', 'local_steps': 0}
# expt = {'topology': 'fully_connected', 'local_steps': 5, 'eval_on_average_model': True}
# expt2 = {'topology': 'fully_connected', 'local_steps': 10}
# expt = {'topology': 'fully_connected', 'local_steps': 50}
# expt = {'topology': 'random', 'degree': 4, 'local_steps': 0}
# expt = {'topology': 'exponential_graph', 'local_steps': 0}
expt = {'topology': 'ring', 'local_steps': 0}
# expt = {'topology': 'ring', 'local_steps': 0, 'data_split': 'no', 'eval_on_average_model': True}

if __name__ == '__main__':

    name = get_expt_name(config, expt)
    wandb.init(name=name, dir='.', config={**config, **expt}, reinit=True, project='MLO-CIFAR10', entity='morales97')
    acc, test_loss, train_loss, _, _, _ = train_cifar(config, expt, wandb)
    wandb.finish()


