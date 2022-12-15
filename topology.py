from cmath import exp
import numpy as np
import pdb
import torch
import copy 

def get_diff_matrix(expt, num_clients):
    topology = expt['topology']

    if topology == 'solo' or num_clients == 1:
        W = np.eye(num_clients)

    elif topology == 'ring':
        if num_clients == 2:
            W = np.ones((2,2))/2
        else:
            W = 1/3 * (np.eye(num_clients) + np.eye(num_clients, k=1) + np.eye(num_clients, k=-1) + np.eye(num_clients, k=num_clients-1) + np.eye(num_clients, k=-num_clients+1)) 
    
    elif topology in ['fully_connected', 'FC_randomized_local_steps']:
        W = np.ones((num_clients, num_clients)) / num_clients

    elif topology == 'FC_alpha':    
        alpha = expt['local_steps'] / (1+expt['local_steps'])
        W = alpha * np.eye(num_clients) + (1-alpha) * np.ones((num_clients,num_clients)) / num_clients
    
    elif topology == 'exponential_graph':
        connections = [2**i for i in range(int(np.log2(num_clients)))]
        W = np.eye(num_clients)/(len(connections)+1)
        for c in connections:
            W += (np.eye(num_clients, k=c) + np.eye(num_clients, k=c-num_clients))/(len(connections)+1)

    elif topology in ['EG_time_varying', 'EG_multi_step', 'EG_time_varying_random']:    
        # EG_time_varying: time-varying exponential graph. Sampled sequentially, one subgraph per step
        # EG_multi_step: same, but execute all subgraphs sequentially at each step: full averaging in less communications
        # EG_time_varying_random: at each step, sample randomly from the list of subgraphs
        connections = [2**i for i in range(int(np.log2(num_clients)))]
        W = [np.eye(num_clients)/2 for _ in connections]
        for i, c in enumerate(connections):
            W[i] += (np.eye(num_clients, k=c) + np.eye(num_clients, k=c-num_clients))/2

    elif topology == 'random':
        W = np.eye((num_clients))
        for i in range(num_clients):
            indxs = np.arange(num_clients).tolist()
            indxs.remove(i)
            edges = np.random.choice(indxs, expt['degree']-1, replace=False)     # NOTE: considering self-connection for degree
            W[i][edges] = 1     # degree is in-degree, out-degree is not controlled. W[edges][i] would make degree in out-degree
        W /= (expt['degree'])

    return W

def diffuse(W, models, step, expt):
    if expt['topology'] == 'centralized':
        return
    
    if expt['topology'] == 'FC_alpha':          # implicit local steps with W
        diffuse_params(W, models)

    elif expt['topology'] == 'FC_randomized_local_steps':     # take a local step with prob (1-p)
        p = 1 / (1+expt['local_steps'])
        if np.random.uniform() < p:
            diffuse_params(W, models)
        else:
            pass

    elif expt['local_steps'] > 0 and (step+1) % expt['local_steps'] != 0:
        pass

    elif isinstance(W, list):
        if 'time_varying' in expt['topology']:      # time-varying
            if expt['topology'] == 'EG_time_varying_random':
                diffuse_params(W[np.random.choice(len(W))], models)
            else:
                diffuse_params(W[step%len(W)], models)
        else:
            for i in range(len(W)):         # multi-step
                diffuse_params(W[i], models)
    else:
        diffuse_params(W, models)


def diffuse_params(W, models):
    """Diffuse the models with their neighbors."""
    models_sd = [copy.deepcopy(model.state_dict()) for model in models]
    keys = models_sd[0].keys()
    for model, weights in zip(models, W):
        neighbors = np.nonzero(weights)[0]
        model.load_state_dict(
            {
                key: torch.stack(
                    [weights[j]*models_sd[j][key] for j in neighbors],
                    dim=0,
                ).sum(0) / weights.sum() 
                for key in keys
            }
        )
