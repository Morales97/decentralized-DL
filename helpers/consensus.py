import numpy as np
import pdb
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from models import get_model

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

def compute_weight_distance(config, model, init_model):
    '''weight distance (L2) between current model and origin'''
    sd = model.state_dict()
    init_sd = init_model.state_dict()
    dist = 0
    for key in sd.keys():
        dist += torch.sum((sd[key] - init_sd[key])**2)
    return torch.sqrt(dist).item()

def compute_node_disagreement_per_layer(config, models, n_nodes):
    ''' Does first layer learn generic features that don't disagree as much? '''
    assert config['net'] == 'convnet'
    # dict_keys(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias'])

    avg_model = get_model(config, 'cpu')
    state_dict_avg = avg_model.state_dict()
    models_sd = [models[i].state_dict() for i in range(n_nodes)]
    L2_diff = {'conv1': 0, 'conv2': 0, 'fc1': 0}
    for key in state_dict_avg.keys():
        state_dict_avg[key] = torch.stack(
                                        [models_sd[i][key] for i in range(n_nodes)], dim=0
                                        ).sum(0) / n_nodes
        for key2 in L2_diff.keys():
            if key2 in key:
                L2_diff[key2] += torch.stack(
                                    [(state_dict_avg[key] - models_sd[i][key])**2 for i in range(n_nodes)], dim=0
                                    ).sum() / n_nodes
    return np.array(list(L2_diff.values()))

def compute_normalized_node_disagreement_per_layer(config, models, n_nodes):
    ''' Does first layer learn generic features that don't disagree as much? '''
    assert config['net'] == 'convnet'
    # dict_keys(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias'])

    avg_model = get_model(config, 'cpu')
    state_dict_avg = avg_model.state_dict()
    models_sd = [models[i].state_dict() for i in range(n_nodes)]
    L2_diff = {'conv1': 0, 'conv2': 0, 'fc1': 0}
    for key in state_dict_avg.keys():
        state_dict_avg[key] = torch.stack(
                                        [models_sd[i][key] for i in range(n_nodes)], dim=0
                                        ).sum(0) / n_nodes
        for key2 in L2_diff.keys():
            if key2 in key:
                L2_diff[key2] += torch.stack(
                                    [(state_dict_avg[key] - models_sd[i][key])**2 for i in range(n_nodes)], dim=0
                                    ).sum()
    
    L2_diff['conv1'] /= torch.flatten(models[0].conv1.weight).shape[0] + torch.flatten(models[0].conv1.bias).shape[0]
    L2_diff['conv2'] /= torch.flatten(models[0].conv2.weight).shape[0] + torch.flatten(models[0].conv2.bias).shape[0]
    L2_diff['fc1'] /= torch.flatten(models[0].fc1.weight).shape[0] + torch.flatten(models[0].fc1.bias).shape[0]
    return np.array(list(L2_diff.values()))