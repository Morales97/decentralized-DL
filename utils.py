import json
import pdb
from numpy import save
import time
import numpy as np
import os

def get_expt_name(config, expt, warmup=False):
    if expt['topology'] == 'centralized' or expt['topology'] == 'fully_connected':
        topology = 'FC' 
    else:
        topology = expt['topology']
    name = topology + '_b' + str(config['batch_size']) + '_lr' + str(config['lr'])

    if warmup:
        name += '_warmup'
        
    return name

def get_folder_name(config, root='experiments_mnist/results/'):
    folder = root
    if config['net'] == 'convnet':
        folder += 'CNN'
    elif config['net'] == 'convnet_op':
        folder += 'CNNOP'
    elif config['net'] == 'mlp':
        folder += 'MLP'
    folder += '_n' + str(config['n_nodes'])
    folder += '_b' + str(config['batch_size'])
    if config['p_label_skew'] > 0:
        folder += '_p' + str(config['p_label_skew'])
    return folder

def get_sweep_filename(config, sweep_of):
    filename = sweep_of + '_SWEEP_'
    filename += config['topology'] 
    filename += '_local' + str(config['local_steps'])
    filename += '_steps' + str(config['steps'])
    if config['data_split'] == 'no':
        filename += '_IID'
    filename += '.json'
    return filename

def save_experiment(config, acc=None, test_loss=None, train_loss=None, consensus=None, path=None, root='experiments_mnist/results/'):
    if path is None:
        # folder
        folder = get_folder_name(config, root)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # filename
        filename = ''
        filename += config['topology'] 
        filename += '_local' + str(config['local_steps'])
        filename += '_lr' + str(np.round(config['lr'],4))
        if config['data_split'] == 'no':
            filename += '_IID'
        filename += '.json'

        path = os.path.join(folder, filename)
        if os.path.exists(path):
            path = path[:-5] + '_time' + str(time.time())[:9] +'.json' # NOTE to break ambiguity and not overwrite. temporal solution
    
    dicts = {'timestamp': time.time(), 'config': config}

    f = open(path, 'w')
    # f.write(json.dumps(config))
    if acc is not None:
        dicts['accuracy'] = acc
    if test_loss is not None:
        dicts['test_loss'] = test_loss
    if train_loss is not None:
        dicts['train_loss'] = train_loss
    if consensus is not None:
        dicts['conensus'] = consensus
    f.write(json.dumps(dicts))
    f.close()

    return path

def save_sweep(config, steps=None, acc=None, test_loss=None, train_loss=None, path=None, root='experiments_mnist/results/', sweep_of='LR'):
    if path is None:
        # folder
        folder = get_folder_name(config, root)

        if not os.path.isdir(folder):
            os.makedirs(folder)

        # filename
        filename = get_sweep_filename(config, sweep_of)

        path = os.path.join(folder, filename)
        if os.path.exists(path):
            path = path[:-5] + '_time' + str(time.time())[:9] +'.json' # NOTE to break ambiguity and not overwrite. temporal solution
    
    dicts = {'timestamp': time.time(), 'config': config}

    if steps is not None:
        dicts['steps'] = steps
    if acc is not None:
        dicts['accuracy'] = acc
    if test_loss is not None:
        dicts['test_loss'] = test_loss
    if train_loss is not None:
        dicts['train_loss'] = train_loss
    
    with open(path, 'w') as f:
        f.write(json.dumps(dicts))

    return path



def load_results(path):
    f = open(path)
    dicts = json.load(f)
    f.close()
    return dicts['accuracy'], dicts['test_loss'], dicts['train_loss']


def load_sweep_results(path):
    f = open(path)
    dicts = json.load(f)
    f.close()
    return dicts['steps'], dicts['accuracy'], dicts['test_loss'], dicts['train_loss']


if __name__ == '__main__':
    # config = {'test': 1}
    # save_experiment(config, None, filename='experiments_mnist/results/test')
    f = open('experiments_mnist/results/test.json')
    dicts = json.load(f)
    f.close()
    pdb.set_trace()