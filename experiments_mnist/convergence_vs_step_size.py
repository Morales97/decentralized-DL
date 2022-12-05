import numpy as np
import sys
import os
import pdb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from mnist_cnn import train_mnist


def sweep_step_sizes(config, expt):
    arr_steps = []
    arr_test_loss = []
    for lr in config['lrs']:
        config['lr'] = lr
        print('*** %s, lr: %.3f ***' % (expt['label'], lr))
        accuracies, test_loss, _ = train_mnist(config, expt)
        steps = len(accuracies)*config['steps_eval']
        if steps == config['steps']:
            break 
        arr_steps.append(steps)
        arr_test_loss.append(test_loss)

    return arr_steps, arr_test_loss

def plot_convergence_vs_lr(ax, steps, lrs, label=None):
    x = lrs[:len(steps)]
    ax.plot(x, steps, label=label)
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Steps until conv')


def plot_test_loss_vs_lr(ax, arr_test_loss, lrs, label=None):
    x = lrs[:len(arr_test_loss)]
    ax.plot(x, arr_test_loss, label=label)
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Test loss')


def plot_all():
    ''' plot epochs till convergence and final test loss '''

    fig, ax = plt.subplots(1,2, figsize=(11,5))
    for i in range(len(expts)):
        arr_steps, arr_test_loss = sweep_step_sizes(config, expts[i])
        plot_convergence_vs_lr(ax[0], arr_steps, config['lrs'], expts[i]['label'])
        plot_test_loss_vs_lr(ax[1], arr_test_loss, config['lrs'], expts[i]['label'])
    plt.legend()
    plt.show()



config = {
    'n_nodes': 15,      # at 15 nodes and batch_size 20, epochs are 200 steps
    'batch_size': 20,
    'lr': 0.1,
    'steps': 1000,
    'steps_eval': 50,  
    'data_split': 'yes',     # NOTE 'no' will sample with replacement from the FULL dataset, which will be truly IID
    'same_init': True,
    'small_test_set': True,
    'lrs': np.logspace(np.log10(0.04),np.log10(1), 7),
    'acc_th': 0.975,
}

expts = [
    {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0},
    # {'topology': 'solo', 'label': 'solo', 'local_steps': 0},
    # {'topology': 'fully_connected', 'label': 'Fully connected', 'local_steps': 0},
    # {'topology': 'fully_connected', 'label': 'FC, 50 local steps', 'local_steps': 50},
    # {'topology': 'fully_connected', 'label': 'FC, 50 local steps, IID', 'local_steps': 50, 'data_split': 'no'},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 580 local steps', 'local_steps': 580},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 580 local steps', 'local_steps': 580, 'data_split': 'no'},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph', 'local_steps': 0},
    # {'topology': 'EG_time_varying', 'label': 'EG time-varying', 'local_steps': 0},
    # {'topology': 'EG_time_varying_random', 'label': 'EG time-varying random', 'local_steps': 0},
    # {'topology': 'EG_multi_step', 'label': 'EG multi-step', 'local_steps': 0},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph, 58 local steps', 'local_steps': 58},
    {'topology': 'random', 'degree': 4, 'label': 'random (degree: 4)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 7, 'label': 'random (degree: 7)', 'local_steps': 0},
    {'topology': 'ring', 'label': 'ring', 'local_steps': 0},

]


if __name__ == '__main__':
    plot_all()
