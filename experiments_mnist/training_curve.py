from xml.parsers.expat import model
import numpy as np
import pdb
import matplotlib.pyplot as plt
import sys
import os
import time 

sys.path.insert(0, os.path.join(sys.path[0], '..'))

from mnist_cnn import train_mnist



def plot_test_accuracy_vs_epoch(config, expt, accuracies):
    epochs = np.arange(config['epochs'])
    assert len(epochs) == len(accuracies)
    
    plt.plot(epochs, accuracies, label=expt['label'])
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')

def plot_train_loss_vs_epoch(config, expt, train_loss):
    epochs = np.arange(config['epochs'])
    # steps = np.arange(config['epochs'] * int(60000 // (config['n_nodes']*config['batch_size'])))
    # assert len(steps) == len(train_loss)
    train_loss = np.array(train_loss)
    train_loss = train_loss.reshape(-1, int(60000 // (config['n_nodes']*config['batch_size'])))
    train_loss_avg = np.mean(train_loss, axis=1)
    assert len(epochs) == len(train_loss_avg)
    # plt.plot(steps, train_loss, label=expt['label'])
    plt.plot(epochs, train_loss_avg, label=expt['label'])
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')

def accuracy_vs_epoch(config, expts):
    for i in range(len(expts)):
        accuracies, loss_test, loss_train = train_mnist(config, expts[i])
        plot_test_accuracy_vs_epoch(config, expts[i], accuracies)
        # plot_train_loss_vs_epoch(config, expts[i], loss_train)
    plt.legend()
    plt.show()

def acc_and_loss_vs_steps(config, expts):
    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Test Accuracy')
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Test Loss')

    for i in range(len(expts)):
        new_config = {**config, **expts[i]} 
        accuracies, loss_test, _ = train_mnist(new_config, expts[i])
        steps = np.arange(1, len(accuracies)+1)*config['steps_eval']
        axes[0].plot(steps, accuracies, label=expts[i]['label'])
        axes[1].plot(steps, loss_test, label=expts[i]['label'])

    plt.legend()
    plt.show()

def node_disagreement(config, expts):
    for i in range(len(expts)):
        new_config = {**config, **expts[i]} 
        accuracies, loss_test, node_disagreement = train_mnist(new_config, expts[i])
        steps = np.arange(len(node_disagreement))
        plt.plot(steps, node_disagreement, label=expts[i]['label'])
    plt.xlabel('Iterations')
    plt.ylabel('Model diff norm')
    plt.legend()
    plt.show()


config = {
    'n_nodes': 15,      # at 15 nodes and batch_size 20, epochs are 200 steps
    'batch_size': 20,
    'lr': 0.1,
    'steps': 100,
    'steps_eval': 100,  
    'data_split': 'yes',     # NOTE 'no' will sample with replacement from the FULL dataset, which will be truly IID
    'same_init': True,
    'small_test_set': True,
}

expts = [
    # {'topology': 'centralized', 'label': 'Fully connected', 'local_steps': 0},
    # {'topology': 'solo', 'label': 'solo', 'local_steps': 0},
    # {'topology': 'fully_connected', 'label': 'Fully connected', 'local_steps': 0},
    {'topology': 'fully_connected', 'label': 'FC, 20 local steps', 'local_steps': 20},
    # {'topology': 'fully_connected', 'label': 'FC, 50 local steps', 'local_steps': 50},
    # {'topology': 'fully_connected', 'label': 'FC, 50 local steps, IID', 'local_steps': 50, 'data_split': 'no'},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 580 local steps', 'local_steps': 580},
    # {'topology': 'fully_connected', 'label': 'Fully connected, 580 local steps', 'local_steps': 580, 'data_split': 'no'},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph', 'local_steps': 0},
    # {'topology': 'EG_time_varying', 'label': 'EG time-varying', 'local_steps': 0},
    # {'topology': 'EG_time_varying_random', 'label': 'EG time-varying random', 'local_steps': 0},
    # {'topology': 'EG_multi_step', 'label': 'EG multi-step', 'local_steps': 0},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph, 58 local steps', 'local_steps': 58},
    # {'topology': 'random', 'degree': 5, 'label': 'random (degree: 5)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 7, 'label': 'random (degree: 7)', 'local_steps': 0},
    # {'topology': 'ring', 'label': 'ring', 'local_steps': 0},

]



if __name__ == '__main__':
    ts = time.time()
    # acc_and_loss_vs_steps(config, expts)
    node_disagreement(config, expts)
    print('** TOTAL TIME: %.2f min **' % ((time.time()-ts)/60))