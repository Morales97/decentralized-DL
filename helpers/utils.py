import json
import pdb
import sched
import shutil
import time
import numpy as np
import os
import torch 

def get_expt_name(args, warmup=False):
    if args.topology[0] == 'centralized' or args.topology[0] == 'fully_connected':
        topology = 'FC' 
    elif args.topology[0] == 'exponential_graph':
        topology = 'EG' 
    else:
        topology = args.topology[0]
    name = topology
    
    if args.n_nodes[0] not in [1, 16]:
        name += '_n' + str(args.n_nodes[0])
    
    name += '_b' + str(args.batch_size) + '_lr' + str(args.lr)

    # if warmup:
    #     name += '_warmup'
    
    if args.local_steps[0] > 0:
        name += '_local' + str(args.local_steps)

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


class AccuracyTracker(object):
    def __init__(self):
        self.max_correct = 0

    def update(self, acc):
        if acc > self.max_correct:
            self.max_correct = acc

    def get(self):
        return self.max_correct

class MultiAccuracyTracker(object):
    def __init__(self, keys):
        self._max_correct = {}
        self._is_best = {}
        for key in keys:
            self._max_correct[key] = 0
            self._is_best[key] = False

    def init(self, keys):
        for key in keys:
            self._max_correct[key] = 0

    def update(self, acc, key):
        if acc > self._max_correct[key]:
            self._max_correct[key] = acc
            self._is_best[key] = True
        else:
            self._is_best[key] = False

    def get(self, key):
        return self._max_correct[key]

    def is_best(self, key):
        return self._is_best[key]

class TrainMetricsTracker(object):
    def __init__(self, keys):
        self._correct = {}
        self._loss = {}
        self._n = {}
        for key in keys:
            self.reset(key)

    def reset(self, key):
        self._correct[key] = 0
        self._loss[key] = 0
        self._n[key] = 0     

    def update(self, correct, loss, n):
        self._correct += correct
        self._loss += loss
        self._n += n

    def get(self, key):
        acc = self._correct / self._n * 100
        loss = self._loss / self._n
        self.reset(key)

        return acc, loss


def save_checkpoint(args, models, ema_models, opts, schedulers, epoch, step, name=None):
    if not isinstance(models, list):
        state = {
            'epoch': epoch,
            'step': step,
            'net': args.net,
            'state_dict': models.state_dict(),
            'ema_state_dict': ema_models[args.alpha[-1]].state_dict(),
            'optimizer' : opts.state_dict(),
            'scheduler': schedulers.state_dict()
        }
        if name:
            torch.save(state, os.path.join(args.save_dir, args.dataset, args.net, args.expt_name, f'checkpoint_{name}.pth.tar'))    
        else:
            torch.save(state, os.path.join(args.save_dir, args.dataset, args.net, args.expt_name, f'checkpoint_{step}.pth.tar'))

    else:
        for i in range(len(models)):
            state = {
                'epoch': epoch,
                'step': step,
                'net': args.net,
                'state_dict': models[i].state_dict(),
                'ema_state_dict': ema_models[args.alpha[-1]][i].state_dict(),
                'optimizer' : opts[i].state_dict(),
                'scheduler': schedulers[i].state_dict()
            }
            if name:
                torch.save(state, os.path.join(args.save_dir, args.dataset, args.net, args.expt_name, f'checkpoint_m{i}_{name}.pth.tar'))    
            else:
                torch.save(state, os.path.join(args.save_dir, args.dataset, args.net, args.expt_name, f'checkpoint_m{i}_{step}.pth.tar'))

            # if args.wandb:
            #     model_artifact = wandb.Artifact('ckpt_m' + str(i), type='model')
            #     model_artifact.add_file(filename=SAVE_DIR + 'checkpoint_m' + str(i) + '.pth.tar')
            #     wandb.log_artifact(model_artifact)
    print('Checkpoint(s) saved!')


if __name__ == '__main__':
    # config = {'test': 1}
    # save_experiment(config, None, filename='experiments_mnist/results/test')
    f = open('experiments_mnist/results/test.json')
    dicts = json.load(f)
    f.close()
    pdb.set_trace()