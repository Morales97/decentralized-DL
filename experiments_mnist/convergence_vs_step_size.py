import numpy as np
import sys
import os
import pdb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from train_mnist_OLD import train_mnist
from helpers.utils import save_experiment, load_results, save_sweep, load_sweep_results, get_sweep_filename, get_folder_name


def sweep_step_sizes(config, expt, early_stop=False, save_expts=False, do_save_sweep=True, root=None):
    arr_steps = []
    arr_accuracy = []
    arr_test_loss = []
    arr_train_loss = []
    for lr in config['lrs']:
        config['lr'] = lr
        print('*** %s, lr: %.3f ***' % (expt['label'], lr))
        accuracies, test_losses, train_losses, _, _, _ = train_mnist(config, expt)
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

    if do_save_sweep:
        if root is None:
            save_sweep(config, arr_steps, arr_accuracy, arr_test_loss, arr_train_loss)
        else:
            save_sweep(config, arr_steps, arr_accuracy, arr_test_loss, arr_train_loss, root=root)
    return arr_steps, arr_accuracy, arr_test_loss, arr_train_loss

def plot_convergence_vs_lr(ax, steps, lrs, label=None):
    x = lrs[:len(steps)]
    ax.plot(x, steps, label=label)
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Steps until conv')

def plot_train_loss_vs_lr(ax, arr_train_loss, lrs, label=None):
    x = lrs[:len(arr_train_loss)]
    ax.plot(x, arr_train_loss, label=label)
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Train loss')
    ax.set_ylim(0, 0.5)

def plot_test_loss_vs_lr(ax, arr_test_loss, lrs, label=None):
    x = lrs[:len(arr_test_loss)]
    ax.plot(x, arr_test_loss, label=label)
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Test loss')
    ax.set_ylim(0, 0.5)

def plot_accuracy_vs_lr(ax, arr_accuracy, lrs, label=None):
    x = lrs[:len(arr_accuracy)]
    ax.plot(x, arr_accuracy, label=label)
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(50, 100)


def plot_all():
    ''' plot epochs till convergence and final test loss '''
    fig, ax = plt.subplots(1,3, figsize=(16,5))
    for i in range(len(expts)):
        new_config = {**config, **expts[i]} 
        arr_steps, arr_accuracy, arr_test_loss, _ = sweep_step_sizes(new_config, expts[i], early_stop=True)
        plot_convergence_vs_lr(ax[0], arr_steps, config['lrs'], expts[i]['label'])
        plot_accuracy_vs_lr(ax[1], arr_accuracy, config['lrs'], expts[i]['label'])
        plot_test_loss_vs_lr(ax[2], arr_test_loss, config['lrs'], expts[i]['label'])
    for i in range(len(ax)):
        ax[i].legend()
    # plt.legend()
    plt.show()

def plot_all_no_threshold(config, expts, load_paths=None, root=None):
    ''' Instead of stopping training at a certain threshold, plot the loss and accuracy after X fixed steps '''

    accuracies = []
    test_losses = []
    train_losses = []
    
    for i in range(len(expts)):
        if load_paths is not None:
            _, arr_accuracy, arr_test_loss, arr_train_loss = load_sweep_results(load_paths[i])
        else:
            new_config = {**config, **expts[i]} 
            _, arr_accuracy, arr_test_loss, arr_train_loss = sweep_step_sizes(new_config, expts[i], root=root)
        accuracies.append(arr_accuracy)
        test_losses.append(arr_test_loss)
        train_losses.append(arr_train_loss)
    
    fig, ax = plt.subplots(1,3, figsize=(16,5))
    for i in range(len(expts)):
        plot_train_loss_vs_lr(ax[0], train_losses[i], config['lrs'], expts[i]['label'])
        plot_accuracy_vs_lr(ax[1], accuracies[i], config['lrs'], expts[i]['label'])
        plot_test_loss_vs_lr(ax[2], test_losses[i], config['lrs'], expts[i]['label'])
    for i in range(len(ax)):
        ax[i].legend()
    # plt.legend()
    plt.show()

def get_sweep_file_paths(config, expts):
    paths = []
    folder = get_folder_name(config)
    for i in range(len(expts)):
        new_config = {**config, **expts[i]}
        filename = get_sweep_filename(new_config)
        paths.append(os.path.join(folder, filename))
    return paths

config = {
    'n_nodes': 15,      # at 15 nodes and batch_size 20, epochs are 200 steps
    'batch_size': 20,
    'steps': 1000, 
    # 'steps': 2000, 
    # 'steps_eval': 2000,  
    'steps_eval': 1000,  
    'data_split': 'yes',     # NOTE 'no' will sample with replacement from the FULL dataset, which will be truly IID
    'same_init': True,
    'lrs': list(np.logspace(np.log10(0.04),np.log10(0.8), 7)),
    'p_label_skew': 0,
    # 'acc_th': 0.975,
    # 'train_loss_th': 0.05,
    # 'train_loss_th': 0.1,
    # 'net': 'mlp',
    'net': 'convnet',
    # 'net': 'convnet_op',
    'eval_on_average_model': True,
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
    {'topology': 'ring', 'label': 'ring', 'local_steps': 0},
    # {'topology': 'ring', 'label': 'ring, IID', 'local_steps': 0, 'data_split': 'no'},

]


if __name__ == '__main__':
    # plot_all()
    plot_all_no_threshold(config, expts)
    # paths = get_sweep_file_paths(config, expts)
    # plot_all_no_threshold(config, expts, paths)