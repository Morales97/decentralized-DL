from platform import node
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

def plot_distance_vs_steps(config, expt, distances):
    steps = np.arange(len(distances))
    
    plt.plot(steps, distances, label=expt['label'])
    plt.xlabel('Steps')
    plt.ylabel('Weight distance to init')

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
        accuracies, loss_test, loss_train, _, _, weight_dist = train_mnist(config, expts[i])
        plot_test_accuracy_vs_epoch(config, expts[i], accuracies)
        # plot_distance_vs_steps(config, expts[i], weight_dist)
        # plot_train_loss_vs_epoch(config, expts[i], loss_train)
    plt.legend()
    plt.show()

def acc_and_loss_vs_steps(config, expts):
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Test Accuracy')
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Test Loss')

    for i in range(len(expts)):
        new_config = {**config, **expts[i]} 
        accuracies, loss_test, _, _, _ = train_mnist(new_config, expts[i])
        steps = np.arange(1, len(accuracies)+1)*config['steps_eval']
        axes[0].plot(steps, accuracies, label=expts[i]['label'])
        axes[1].plot(steps, loss_test, label=expts[i]['label'])

    plt.legend()
    plt.show()

def node_disagreement(config, expts):
    for i in range(len(expts)):
        new_config = {**config, **expts[i]} 
        accuracies, loss_test, _, node_disagreement, _ = train_mnist(new_config, expts[i])
        steps = np.arange(len(node_disagreement))
        plt.plot(steps, node_disagreement, label=expts[i]['label'])
    plt.xlabel('Iterations')
    plt.ylabel('Model diff norm')
    plt.legend()
    plt.show()

def node_disagreement_per_layer(config, expts):
    for i in range(len(expts)):
        new_config = {**config, **expts[i]} 
        accuracies, loss_test, _, node_disagreement, _ = train_mnist(new_config, expts[i])
        node_disagreement = np.array(node_disagreement)
        steps = np.arange(node_disagreement.shape[0])
        plt.plot(steps, node_disagreement[:,0], label=expts[i]['label']+'_conv1')
        plt.plot(steps, node_disagreement[:,1], label=expts[i]['label']+'_conv2')
        plt.plot(steps, node_disagreement[:,2], label=expts[i]['label']+'_fc1')
    plt.xlabel('Iterations')
    plt.ylabel('Model diff norm')
    plt.legend()
    plt.show()

def plot_all(config, expts):
    accuracies = []
    test_losses = []
    train_losses = []
    distances = []
    steps = []
    consensus = []
    for i in range(len(expts)):
        new_config = {**config, **expts[i]} 
        acc, loss, train_loss, node_disagreement, _, weight_dist = train_mnist(new_config, expts[i])
        s = np.arange(1, len(acc)+1)*config['steps_eval']
        accuracies.append(acc)
        test_losses.append(loss)
        train_losses.append(train_loss)
        distances.append(weight_dist)
        steps.append(s)
        consensus.append(node_disagreement)
    
    fig, axes = plt.subplots(2, 2, figsize=(13,7), dpi=100)
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Test Loss')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Train Loss')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Weight distance to init')

    for i in range(len(expts)):
        axes[0, 0].plot(steps[i], accuracies[i], label=expts[i]['label'])
        axes[0, 1].plot(steps[i], test_losses[i], label=expts[i]['label'])
        train_l = np.array(train_losses[i]).reshape(10, -1).mean(axis=0)
        axes[1, 0].plot(np.arange(len(train_l))*10, train_l, label=expts[i]['label'])
        axes[1, 1].plot(np.arange(len(distances[i]))*25, distances[i], label=expts[i]['label'])
    
    plt.legend()
    plt.show()


def gradient_variance(config, expts):
    # fig, axes = plt.subplots(1, 3, figsize=(17,5))
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    for i in range(len(expts)):
        new_config = {**config, **expts[i]} 
        accuracies, loss_test, _, node_disagreement, gradient_variance = train_mnist(new_config, expts[i])
        gradient_variance = np.array(gradient_variance)
        steps = np.arange(1,len(gradient_variance)+1)*config['steps_grad_var']
        
        axes[0].plot(steps, gradient_variance[:,0], label=expts[i]['label']+' - Ratio')
        axes[1].plot(steps, gradient_variance[:,1], label=expts[i]['label']+' - Stochastic')
        axes[1].plot(steps, gradient_variance[:,2], label=expts[i]['label']+' - Node')
        # axes[2].plot(np.arange(len(node_disagreement)), node_disagreement, label=expts[i]['label'])
    axes[0].set_xlabel('Iterations')
    axes[1].set_xlabel('Iterations')
    # axes[2].set_xlabel('Iterations')
    axes[0].set_ylabel('Ratio')
    axes[1].set_ylabel('Variance')
    # axes[2].set_ylabel('Consensus')
    axes[0].legend()
    axes[1].legend()
    # axes[2].legend()
    plt.show()

def gradient_variance2(config, expts):
    fig, axes = plt.subplots(1, 3, figsize=(17,5))
    for i in range(len(expts)):
        print('***** %s *****' % expts[i]['label'])
        new_config = {**config, **expts[i]} 
        accuracies, loss_test, _, node_disagreement, gradient_variance = train_mnist(new_config, expts[i])
        gradient_variance = np.array(gradient_variance)
        steps = np.arange(1,len(gradient_variance)+1)*config['steps_grad_var']
        
        axes[0].plot(steps, gradient_variance[:,0], label=expts[i]['label']+' - Ratio')
        axes[1].plot(steps, gradient_variance[:,1], label=expts[i]['label']+' - Stochastic')
        axes[2].plot(steps, gradient_variance[:,2], label=expts[i]['label']+' - Node')
    axes[0].set_xlabel('Iterations')
    axes[1].set_xlabel('Iterations')
    axes[2].set_xlabel('Iterations')
    axes[0].set_ylabel('Ratio')
    axes[1].set_ylabel('Stochastic Variance')
    axes[2].set_ylabel('Node variance')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    plt.show()

config = {
    'n_nodes': 15,      # at 15 nodes and batch_size 20, epochs are 200 steps
    'batch_size': 16,
    'lr': 0.15,
    'steps': 2000,
    'steps_eval': 200, #50,  
    'steps_weight_distance': 25,
    'data_split': 'yes',     # NOTE 'no' will sample with replacement from the FULL dataset, which will be truly IID
    'same_init': True,
    'small_test_set': True,
    'p_label_skew': 0,
    'net': 'convnet',
    # 'freeze_step': 200,
    'eval_on_average_model': True,
}

expts = [
    {'topology': 'centralized', 'label': 'Centralized, LR warm up (100)', 'local_steps': 0, 'warmup_steps': 100},
    {'topology': 'centralized', 'label': 'Centralized', 'local_steps': 0},
    # {'topology': 'centralized', 'label': 'Fully connected, sample with replacement', 'local_steps': 0},
    # {'topology': 'solo', 'label': 'solo', 'local_steps': 0},
    # {'topology': 'fully_connected', 'label': 'Fully connected', 'local_steps': 0},
    # {'topology': 'fully_connected', 'label': 'FC, 20 local steps', 'local_steps': 20},
    # {'topology': 'fully_connected', 'label': 'FC, 50 local steps', 'local_steps': 50},
    # {'topology': 'fully_connected', 'label': 'FC, 500 local steps', 'local_steps': 500},
    # {'topology': 'fully_connected', 'label': 'FC, 50 local steps, IID no split', 'local_steps': 50, 'data_split': 'no'},
    # {'topology': 'fully_connected', 'label': 'FC, 500 local steps', 'local_steps': 500},
    # {'topology': 'fully_connected', 'label': 'FC, 1000 local steps', 'local_steps': 1000},
    # {'topology': 'fully_connected', 'label': 'FC, 200 local steps', 'local_steps': 200},
    # {'topology': 'fully_connected', 'label': 'FC, 200 local steps and freeze', 'local_steps': 200, 'freeze_step': 200},
    # {'topology': 'fully_connected', 'label': 'FC, 200 local steps, IID no split', 'local_steps': 200, 'data_split': 'no'},
    # {'topology': 'fully_connected', 'label': 'FC, 600 local steps', 'local_steps': 600},
    # {'topology': 'fully_connected', 'label': 'FC, 600 local steps, IID no split', 'local_steps': 600, 'data_split': 'no'},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph', 'local_steps': 0},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph, IID (no split)', 'local_steps': 0, 'data_split': 'no'},
    # {'topology': 'EG_time_varying', 'label': 'EG time-varying', 'local_steps': 0},
    # {'topology': 'EG_time_varying_random', 'label': 'EG time-varying random', 'local_steps': 0},
    # {'topology': 'EG_multi_step', 'label': 'EG multi-step', 'local_steps': 0},
    # {'topology': 'exponential_graph', 'label': 'Exponential graph, 58 local steps', 'local_steps': 58},
    # {'topology': 'random', 'degree': 4, 'label': 'random (degree: 4)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 5, 'label': 'random (degree: 5)', 'local_steps': 0},
    # {'topology': 'random', 'degree': 7, 'label': 'random (degree: 7)', 'local_steps': 0},
    # {'topology': 'ring', 'label': 'ring, batch: 4', 'local_steps': 0, 'batch_size': 4},
    # {'topology': 'ring', 'label': 'ring, batch: 16', 'local_steps': 0, 'batch_size': 16},
    # {'topology': 'ring', 'label': 'ring, batch: 32', 'local_steps': 0, 'bathc_size': 32},
    {'topology': 'ring', 'label': 'ring', 'local_steps': 0},
    # {'topology': 'ring', 'label': 'ring, lr: 0.1, p=0.5', 'local_steps': 0, 'p_label_skew': 0.5},
    # {'topology': 'ring', 'label': 'ring, lr: 0.1, p=1', 'local_steps': 0, 'p_label_skew': 1},
    # {'topology': 'ring', 'label': 'ring, lr: 0.07', 'local_steps': 0, 'lr': 0.07},
    # {'topology': 'ring', 'label': 'ring, lr: 0.04', 'local_steps': 0, 'lr': 0.04},
    # {'topology': 'ring', 'label': 'ring, lr: 0.8', 'local_steps': 0, 'lr': 0.8},
    # {'topology': 'ring', 'label': 'ring, lr: 0.01', 'local_steps': 0, 'lr': 0.01},
    # {'topology': 'ring', 'label': 'ring, lr: 0.5', 'local_steps': 0, 'lr': 0.5},
    # {'topology': 'ring', 'label': 'ring, lr: 0.05', 'local_steps': 0, 'lr': 0.05},
    # {'topology': 'ring', 'label': 'ring, p=1', 'local_steps': 0, 'p_label_skew': 1},
    # {'topology': 'ring', 'label': 'ring, IID (no split)', 'local_steps': 0, 'data_split': 'no'},
]


if __name__ == '__main__':
    ts = time.time()
    # acc_and_loss_vs_steps(config, expts)
    # node_disagreement_per_layer(config, expts)
    plot_all(config, expts)
    # gradient_variance(config, expts)
    # gradient_variance2(config, expts)
    #accuracy_vs_epoch(config, expts)
    print('** TOTAL TIME: %.2f min **' % ((time.time()-ts)/60))