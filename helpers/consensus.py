
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from topology import get_average_model

def compute_node_consensus(args, device, models):
    avg_model = get_average_model(args, device, models)
    state_dict_avg = avg_model.state_dict()
    models_sd = [models[i].state_dict() for i in models]
    L2_diff = 0
    for key in state_dict_avg.keys():
        L2_diff += torch.stack(
                            [(state_dict_avg[key] - models_sd[i][key])**2 for i in range(args.n_nodes)], dim=0
                            ).sum() / args.n_nodes
    return L2_diff