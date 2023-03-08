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

def get_folder_name(args):
    if args.local_exec:
        return os.getcwd()
    return os.path.join(args.save_dir, args.dataset, args.net, args.expt_name + '_s' + str(args.seed))


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
    ''''
    Track best test accuracy and loss, and wheter current model is best
    '''
    def __init__(self, keys):
        self._max_acc = {}
        self._min_loss = {}
        self._is_best_acc = {}
        self._is_best_loss = {}
        self._init(keys)

    def _init(self, keys):
        for key in keys:
            self._max_acc[key] = 0
            self._min_loss[key] = 1e5
            self._is_best_acc[key] = False
            self._is_best_loss[key] = False

    def update(self, acc, loss, key):
        # acc
        if acc > self._max_acc[key]:
            self._max_acc[key] = acc
            self._is_best_acc[key] = True
        else:
            self._is_best_acc[key] = False
        
        # loss
        if loss < self._min_loss[key]:
            self._min_loss[key] = loss
            self._is_best_loss[key] = True
        else:
            self._is_best_loss[key] = False

    def get_acc(self, key):
        return self._max_acc[key]

    def get_loss(self, key):
        return self._min_loss[key]

    def is_best_acc(self, key):
        return self._is_best_acc[key]

    def is_best_loss(self, key):
        return self._is_best_loss[key]

class TrainMetricsTracker(object):
    def __init__(self, keys):
        self._correct = {}
        self._loss = {}
        self._n = {}
        for key in keys:
            self._reset(key)

    def _reset(self, key):
        self._correct[key] = 0
        self._loss[key] = 0
        self._n[key] = 0     

    def update(self, key, correct, loss):
        self._correct[key] += correct
        self._loss[key] += loss
        self._n[key] += 1

    def get(self, key):
        acc = self._correct[key] / self._n[key] * 100
        loss = self._loss[key] / self._n[key]
        self._reset(key)

        return acc, loss


def save_checkpoint(args, models, ema_models, opts, schedulers, epoch, step, name=None, best_alpha=None):
    if args.local_exec: return

    if not isinstance(models, list):
        state = {
            'epoch': epoch,
            'step': step,
            'net': args.net,
            'state_dict': models.state_dict(),
            'optimizer' : opts.state_dict(),
            'scheduler': schedulers.state_dict(),
        }
        for alpha in ema_models.keys():
            state['ema_state_dict_' + str(alpha)] = ema_models[alpha].state_dict(),
        if best_alpha:
            state['best_alpha'] = best_alpha
        if name:
            torch.save(state, os.path.join(get_folder_name(args), f'checkpoint_{name}.pth.tar'))    
        else:
            torch.save(state, os.path.join(get_folder_name(args), f'checkpoint_{step}.pth.tar'))

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
                torch.save(state, os.path.join(get_folder_name(args), f'checkpoint_m{i}_{name}.pth.tar'))    
            else:
                torch.save(state, os.path.join(get_folder_name(args), f'checkpoint_m{i}_{step}.pth.tar'))

            # if args.wandb:
            #     model_artifact = wandb.Artifact('ckpt_m' + str(i), type='model')
            #     model_artifact.add_file(filename=SAVE_DIR + 'checkpoint_m' + str(i) + '.pth.tar')
            #     wandb.log_artifact(model_artifact)
    print('Checkpoint(s) saved!')

def copy_checkpoint(args, ckpt_name='checkpoint_last.pth.tar', new_name='model_best.pth.tar'):
    path = os.path.join(get_folder_name(args))
    ckpt_file = os.path.join(path, ckpt_name)
    new_ckpt_file = os.path.join(path, new_name)
    shutil.copyfile(ckpt_file, new_ckpt_file)

if __name__ == '__main__':
    # config = {'test': 1}
    # save_experiment(config, None, filename='experiments_mnist/results/test')
    f = open('experiments_mnist/results/test.json')
    dicts = json.load(f)
    f.close()
    pdb.set_trace()