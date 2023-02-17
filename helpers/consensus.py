
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import pdb

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from topology import get_average_model

def compute_node_consensus(args, device, models):
    '''weight distance (L2) between all models'''
    avg_model = get_average_model(device, models)
    state_dict_avg = avg_model.state_dict()
    models_sd = [model.state_dict() for model in models]
    L2_diff = 0
    for key in state_dict_avg.keys():
        if 'weight' in key or 'bias' in key:
            L2_diff += torch.stack(
                                [(state_dict_avg[key] - models_sd[i][key])**2 for i in range(args.n_nodes[0])], dim=0
                                ).sum() / args.n_nodes[0]
    return L2_diff

def compute_weight_distance(model, init_model):
    '''weight distance (L2) between current model and origin'''
    sd = model.state_dict()
    init_sd = init_model.state_dict()
    dist = 0
    for key in sd.keys():
        if 'weight' in key or 'bias' in key:
            dist += torch.sum((sd[key] - init_sd[key])**2)
    return torch.sqrt(dist).item()

def compute_weight_norm(model):
    '''return L2 norm of parameters (as if they where vectorized)'''
    sd = model.state_dict()
    L2_norm = 0
    for key in sd.keys():
        if 'weight' in key or 'bias' in key:
            L2_norm += torch.sum((sd[key])**2)
    return torch.sqrt(L2_norm).item()

def get_gradient_norm(model):
    """ computes gradient norm of parameters, as if all parameters where concatenated in a single vector"""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in grads]))
    return total_norm

def get_momentum_norm(opt):
    moms = []
    for group in opt.param_groups:
        for p in group['params']:
            state = opt.state[p]
            if 'momentum_buffer' in state:
                moms.append(state['momentum_buffer'])

    return torch.norm(torch.stack([torch.norm(m) for m in moms]))

def get_grad_stats(opt):
    '''
    Did not observe anything meaningful in the stats of gradient. Mean is always around 0 and grad**2 is always close to 0
    (Pdb) np.histogram(means)
    (array([    2,     9,    94, 31910,  2726,    59,    18,     6,     1,
           1]), array([-6.3870696e-04, -4.7865612e-04, -3.1860528e-04, -1.5855445e-04,
        1.4963792e-06,  1.6154721e-04,  3.2159805e-04,  4.8164889e-04,
        6.4169971e-04,  8.0175058e-04,  9.6180139e-04], dtype=float32))
    (Pdb) np.histogram(vars)
    (array([34788,    22,     8,     3,     3,     1,     0,     0,     0,
           1]), array([6.8967710e-21, 9.2506189e-06, 1.8501238e-05, 2.7751857e-05,
       3.7002475e-05, 4.6253095e-05, 5.5503715e-05, 6.4754335e-05,
       7.4004951e-05, 8.3255574e-05, 9.2506190e-05], dtype=float32))
    '''
    means = []
    vars = []
    for group in opt.param_groups:
        for p in group['params']:
            state = opt.state[p]
            if 'exp_avg' in state:
                means.append(state['exp_avg'].flatten())
            if 'exp_avg_sq' in state:
                vars.append(state['exp_avg_sq'].flatten())
    means = torch.cat(means)
    vars = torch.cat(vars)
    pdb.set_trace()
