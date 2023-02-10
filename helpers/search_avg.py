import torch
import torch.nn.functional as F
import numpy as np
import pdb

import sys
import os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from helpers.parser import parse_args
from loaders.data import get_data
from helpers.avg_index import UniformAvgIndex, ModelAvgIndex
from model.model import get_model

@torch.no_grad()
def update_bn(loader, model, device=None):
    '''
    From https://github.com/timgaripov/swa
    Perform a pass over loader to recompute BN stats
    BN momentum is set to None -> perform cumulative moving average, not running average
    '''
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def eval_avg_model(model, train_loader, test_loader):
    '''
    Get model average between [start, end]
    Update BN stats on train data
    Evaluate on test data
    '''
    # print('Evaluating average model from %d to %d...' % (start, end))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # update BN stats
    update_bn(train_loader, model, device)

    # evaluate 
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset) * 100

    return loss, acc

def get_exp_idxs(start, min, avl_ckpts):
    ''' get indices exponentially spaced'''
    start_idx = avl_ckpts.index(start)
    min_idx = avl_ckpts.index(min)
    idxs = []

    for i in range(int(np.log2(start_idx))+1):
        if start_idx - 2**i <= min_idx:  
            break
        idxs.append(start_idx - 2**i)

    if min_idx not in idxs:
        idxs.append(min_idx)
    return idxs

def get_exp_idxs_set(start, min, ckpt_period):
    ''' get indices exponentially spaced'''
    start_idx = start // ckpt_period
    min_idx = min // ckpt_period  
    idxs = []

    for i in range(int(np.log2(start_idx))+1):
        if start_idx - 2**i <= min_idx:  
            break
        idxs.append((start_idx - 2**i) * ckpt_period)

    if min_idx not in idxs:
        idxs.append(min_idx * ckpt_period)
    return idxs


def exponential_search(index, train_loader, test_loader, end, start, min=0, accs={}, test=True):
    '''
    recursively search optimal averaging window in log(n) time
    args:
        end: steps (e.g., 5000)
        start: start evaluating avg model between [start, end]. Search models prior to that. (e.g., 4000)
        min: minimum number of steps for to start average (e.g., 2000)
        e.g., will find optimal averaging between [2000-4000, 5000] steps

        accs: previously computed accuracies.
    '''
    if not test:                                    # TODO remove after checking correct behavior
        avl_ckpts = index._index.available_checkpoints
    else:
        avl_ckpts = [1000*i for i in range(50)]
        score = lambda x: 1/((x-20)**2+1e-6)

    accs = accs
    assert start in avl_ckpts, 'non-existing start checkpoint'
    assert end in avl_ckpts, 'non-existing end checkpoint'

    # eval start model
    if not test:                                    # TODO remove after checking correct behavior
        model = index.avg_from(start, until=end)
        _, acc = eval_avg_model(model, train_loader, test_loader)
    else:
        acc = score(avl_ckpts.index(start))
        print('score %.8f at step %d' % (acc, start))
    accs[start] = acc
    best_acc = acc
    best_key = start
    
    # get indexes to search
    if not test:
        search_idxs = get_exp_idxs_set(start, min, index._index._checkpoint_period)
    else:
        # search_idxs = get_exp_idxs(start, min, avl_ckpts) * 1000
        search_idxs = get_exp_idxs_set(start, min, 1000)
    print(f'Searching in {search_idxs}')

    for i, idx in enumerate(search_idxs):
        if idx not in accs.keys():
            if not test:                             # TODO remove after checking correct behavior
                model = index.avg_from(idx, until=end)
                _, acc = eval_avg_model(model, train_loader, test_loader)
                print(f'Acc: {acc} at step {idx}')
            else:
                acc = score(idx//1000)
                print('score %.8f at step %d' % (acc, idx))

            accs[idx] = acc
        if accs[idx] < best_acc:
            break
        else:
            best_acc = acc
            best_key = idx

    pdb.set_trace()
    if best_key == start:
        return best_acc, start
    if best_key == min(avl_ckpts):                    # TODO check logic
        return best_acc, 0
    return exponential_search(index, train_loader, test_loader, end, search_idxs[i-2], min=search_idxs[i], accs=accs)

if __name__ == '__main__':
    ''' For debugging purposes '''
    args = parse_args()

    train_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction)
    train_loader = train_loader[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = os.path.join(args.save_dir, args.expt_name)
    step = 38800 # TODO search in save_dir and get latest index_{step}.pt
    state_dir = os.path.join(save_dir, f'index_{step}.pt')

    uniform_index = UniformAvgIndex('.')
    uniform_index.load_state_dict(state_dir=state_dir)

    index = ModelAvgIndex(
            get_model(args, device),              # NOTE only supported with solo mode now.
            uniform_index,
            include_buffers=True,
        )

    # compute all accuracies in advance and store
    accs = {}
    for i in range(1, 3): #38400//400):
        model = index.avg_from(i*400, until=38400)
        _, acc = eval_avg_model(model, train_loader, test_loader)
        accs[i*400] = acc
    torch.save(accs, os.path.join(save_dir, 'accs_computed.pt'))
    pdb.set_trace()
    exponential_search(index, train_loader, test_loader, end=38400, start=38000, test=False)

# python helpers/search_avg.py 