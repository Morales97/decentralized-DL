import numpy as np
import sys
import os
import pdb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from train_mnist_OLD import train_mnist
from helpers.utils import save_experiment, load_results, save_sweep, load_sweep_results, get_sweep_filename, get_folder_name


def sweep_batch_sizes(config, expt, early_stop=False, save_expts=False, do_save_sweep=True, root=None):
    arr_steps = []
    arr_accuracy = []
    arr_test_loss = []
    arr_train_loss = []
    arr_train_losses = []
    arr_dist = []
    config['BS'] = [int(x) for x in config['BS']]
    for bs in config['BS']:
        config['batch_size'] = bs
        # config['steps'] = int(config['steps_base'] // bs - config['steps_base'] // bs % 5)  # steps multiple of 5
        # config['steps'] = int(config['steps_base'] // np.sqrt(bs) - config['steps_base'] // np.sqrt(bs) % 5)  # steps multiple of 5
        config['steps'] = 1000
        config['steps_eval'] = config['steps']
        # config['lr'] = config['lr_base'] * bs/20 
        # config['lr'] = config['lr_base'] * np.sqrt(bs/20)
        print('*** %s, lr: %.3f, bs: %d ***' % (expt['label'], config['lr'], bs))
        accuracies, test_losses, train_losses, _, _, weight_dist = train_mnist(config, expt)
        if save_expts:
            save_experiment(config, accuracies, test_losses, train_losses)

        steps = len(train_losses)
        if steps == config['steps'] and early_stop:
            break 
        arr_steps.append(steps)
        arr_accuracy.append(accuracies[-1])
        arr_test_loss.append(test_losses[-1])
        train_loss = np.array(train_losses[-25:]).mean()
        arr_train_loss.append(train_loss)
        smoothened_train_curve = np.convolve(np.array(train_losses).reshape(5, -1).mean(axis=0), np.ones(15), mode='valid')
        arr_train_losses.append(smoothened_train_curve)
        arr_dist.append(weight_dist)

    if do_save_sweep:
        if root is None:
            save_sweep(config, arr_steps, arr_accuracy, arr_test_loss, arr_train_loss, sweep_of='BS')
        else:
            save_sweep(config, arr_steps, arr_accuracy, arr_test_loss, arr_train_loss, root=root, sweep_of='BS')
    return arr_steps, arr_accuracy, arr_test_loss, arr_train_loss, arr_train_losses, arr_dist

def plot_convergence_vs_bs(ax, steps, BS, label=None):
    x = BS[:len(steps)]
    ax.plot(x, steps, label=label)
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Steps until conv')

def plot_train_loss_vs_bs(ax, arr_train_loss, BS, label=None):
    x = BS[:len(arr_train_loss)]
    ax.plot(x, arr_train_loss, label=label)
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Train loss')
    ax.set_ylim(0, 0.5)

def plot_test_loss_vs_bs(ax, arr_test_loss, BS, label=None):
    x = BS[:len(arr_test_loss)]
    ax.plot(x, arr_test_loss, label=label)
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Test loss')
    ax.set_ylim(0, 0.3)

def plot_accuracy_vs_bs(ax, arr_accuracy, BS, label=None):
    x = BS[:len(arr_accuracy)]
    ax.plot(x, arr_accuracy, label=label)
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(80, 100)

def plot_train_loss_curve(ax, arr_train_loss, BS, label=None):
    for i in range(len(arr_train_loss)):
        ax.plot(np.arange(len(arr_train_loss[i]))*5, arr_train_loss[i], label=label+'_'+str(BS[i]))
        ax.set_xlabel('Steps')
        ax.set_ylabel('Train loss')

def plot_distance_vs_bs(ax, arr_dist, BS, label=None):
    for i in range(len(arr_dist)):
        ax.plot(np.arange(len(arr_dist[i]))*5, arr_dist[i], label=label+'_'+str(BS[i]))
        ax.set_xlabel('Steps')
        ax.set_ylabel('Weight distance from init')


def plot_all_BS(config, expts, load_paths=None, root=None):
    accuracies = []
    test_losses = []
    train_losses = []
    all_train_losses = []
    distances = []

    for i in range(len(expts)):
        if load_paths is not None:
            _, arr_accuracy, arr_test_loss, arr_train_loss = load_sweep_results(load_paths[i])
        else:
            new_config = {**config, **expts[i]} 
            _, arr_accuracy, arr_test_loss, arr_train_loss, arr_train_losses, arr_dist = sweep_batch_sizes(new_config, expts[i], root=root)
        accuracies.append(arr_accuracy)
        test_losses.append(arr_test_loss)
        train_losses.append(arr_train_loss)
        all_train_losses.append(arr_train_losses)
        distances.append(arr_dist)

    fig, ax = plt.subplots(2,3, figsize=(13,8), dpi=100)
    fig.tight_layout(pad=4)
    for i in range(len(expts)):
        plot_train_loss_vs_bs(ax[0,0], train_losses[i], config['BS'], expts[i]['label'])
        plot_accuracy_vs_bs(ax[0,1], accuracies[i], config['BS'], expts[i]['label'])
        plot_test_loss_vs_bs(ax[0,2], test_losses[i], config['BS'], expts[i]['label'])
        plot_train_loss_curve(ax[1,0], all_train_losses[i], config['BS'], expts[i]['label'])
        plot_distance_vs_bs(ax[1,1], distances[i], config['BS'], expts[i]['label'])
    # for i in range(len(ax)):
    for cax in ax.ravel():
        cax.legend()
    # plt.legend()
    plt.show()

def get_sweep_file_paths(config, expts):
    paths = []
    folder = get_folder_name(config)
    for i in range(len(expts)):
        new_config = {**config, **expts[i]}
        filename = get_sweep_filename(new_config, sweep_of='BS')
        paths.append(os.path.join(folder, filename))
    return paths

config = {
    'n_nodes': 15,      # at 15 nodes and batch_size 20, epochs are 200 steps
    'batch_size': 20,
    'BS': 2**np.arange(1, 8),
    'steps_base': 7000, 
    'data_split': 'yes',     # NOTE 'no' will sample with replacement from the FULL dataset, which will be truly IID
    'same_init': True,
    'lr': 0.5,
    'lr_base': 0.1,
    # 'lrs': list(np.logspace(np.log10(0.04),np.log10(0.8), 7)),
    'p_label_skew': 0,
    'net': 'convnet',
    'eval_on_average_model': True,
    'steps_weight_distance': 25,
}

expts = [
    {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0},
    # {'topology': 'centralized', 'label': 'FC, IID', 'local_steps': 0, 'data_split': 'no'},
    # {'topology': 'fully_connected', 'label': 'Fully connected', 'local_steps': 0},
    # {'topology': 'solo', 'label': 'solo', 'local_steps': 0},
    # {'topology': 'solo', 'label': 'solo, IID', 'local_steps': 0, 'data_split': 'no'},
    # {'topology': 'fully_connected', 'label': 'FC, 100 local steps', 'local_steps': 100},
    # {'topology': 'fully_connected', 'label': 'FC, 100 local steps, IID', 'local_steps': 100, 'data_split': 'no'},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 500 local steps', 'local_steps': 500},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 580 local steps', 'local_steps': 580, 'data_split': 'no'},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph', 'local_steps': 0},
    # {'topology': 'EG_time_varying', 'label': 'EG time-varying', 'local_steps': 0},
    # {'topology': 'EG_time_varying_random', 'label': 'EG time-varying random', 'local_steps': 0},
    # {'topology': 'EG_multi_step', 'label': 'EG multi-step', 'local_steps': 0},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph, 58 local steps', 'local_steps': 58},
    # {'topology': 'random', 'degree': 4, 'label': 'random (degree: 4)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 4, 'label': 'random (degree: 4), IID', 'local_steps': 0, 'data_split': 'no'},
    # {'topology': 'random', 'degree': 7, 'label': 'random (degree: 7)', 'local_steps': 0},
    # {'topology': 'ring', 'label': 'ring', 'local_steps': 0},
    # {'topology': 'ring', 'label': 'ring, IID', 'local_steps': 0, 'data_split': 'no'},

]


if __name__ == '__main__':
    # plot_all()
    plot_all_BS(config, expts)
    # paths = get_sweep_file_paths(config, expts)
    # plot_all_BS(config, expts, paths)