import torch
import torch.nn.functional as F
import numpy as np
import pdb

import sys
import os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from helpers.parser import parse_args
from loaders.data import get_data, ROOT_CLUSTER
from loaders.cifar import get_cifar_filtered_samples
from avg_index.avg_index import UniformAvgIndex, ModelAvgIndex, TriangleAvgIndex
from model.model import get_model

@torch.no_grad()
def update_bn(loader, model, device=None, compute_train_acc=False):
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

    correct = 0
    for input, target in loader:
        input = input.to(device)

        output = model(input)
        
        if compute_train_acc:
            target = target.to(device)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

    if compute_train_acc:
        print(f'Train accuracy: {correct/len(loader.dataset)}')

def eval_avg_model(model, train_loader, test_loader, compute_train_acc=False):
    '''
    Update BN stats on train data
    Evaluate on test data
    '''
    # print('Evaluating average model from %d to %d...' % (start, end))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # update BN stats
    update_bn(train_loader, model, device, compute_train_acc)

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

def three_split_search(index, train_loader, test_loader, end, start, min=0, accs={}, test=True):
    '''
    3-split search. 
    [WARNING] Assumes concavity of accuracy along average models
    '''
    avl_ckpts = list(index._index.available_checkpoints)
    avl_ckpts.sort()

    assert start in avl_ckpts, 'non-existing start checkpoint'
    assert end in avl_ckpts, 'non-existing end checkpoint'
    search_range = np.array(avl_ckpts[avl_ckpts.index(start) : avl_ckpts.index(end)])   # select search range

    while True:
        print(search_range)
        if len(search_range) > 4:
            idxs = [0, len(search_range)//3, 2*len(search_range)//3, len(search_range)-1]
            search_steps = search_range[idxs]
            max_acc = 0
            for step in search_steps:
                if step not in accs.keys():
                    model = index.avg_from(step, until=end)
                    _, acc = eval_avg_model(model, train_loader, test_loader)
                    accs[step] = acc
                max_acc = max(max_acc, accs[step])
                print(f'Acc: {accs[step]} at step {step}')
                
            if accs[search_steps[0]] == max_acc:
                search_range = search_range[:idxs[1]]
            if accs[search_steps[1]] == max_acc:
                search_range = search_range[:idxs[2]]
            if accs[search_steps[2]] == max_acc:
                search_range = search_range[idxs[1]:]
            if accs[search_steps[3]] == max_acc:
                search_range = search_range[idxs[2]:]
        else:
            for step in search_range:
                max_acc = 0
                if step not in accs.keys():
                    model = index.avg_from(step, until=end)
                    _, acc = eval_avg_model(model, train_loader, test_loader)
                    accs[step] = acc
                if max_acc < accs[step]:
                    max_acc = accs[step]
                    max_step = step
            return max_acc, max_step



def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]
def find_index_ckpt(rootdir=".", prefix='index'):
    files = [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.startswith(prefix)
    ]
    file = files[0]
    end_step = file.split('_')[-1].split('.')[0]
    return file, end_step

def get_avg_model(args, start=0.5, end=1):
    '''
    Get an average model for expt_name run between epochs [total_epochs * start, total_epochs * end] 
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction)
    save_dir = os.path.join(args.save_dir, args.expt_name)
    index_ckpt_file, step = find_index_ckpt(save_dir)
    state_dir = os.path.join(save_dir, index_ckpt_file)

    _index = UniformAvgIndex('.')
    state_dict = torch.load(state_dir)
    _index.load_state_dict(state_dict)

    index = ModelAvgIndex(
            get_model(args, device),              # NOTE only supported with solo mode now.
            _index,
            include_buffers=True,
        )
    
    av_ckpts = list(state_dict['available_checkpoints'])
    av_ckpts.sort()
    model = index.avg_from(av_ckpts[int(len(av_ckpts)*start)-1], until=av_ckpts[int(len(av_ckpts)*end)-1])  
    update_bn(train_loader[0], model, device)
    
    return model

if __name__ == '__main__':
    ''' For debugging purposes '''
    args = parse_args()

    train_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction)
    train_loader = train_loader[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = os.path.join(args.save_dir, args.expt_name)
    index_ckpt_file, step = find_index_ckpt(save_dir)
    state_dir = os.path.join(save_dir, index_ckpt_file)

    _index = UniformAvgIndex('.')
    # _index = TriangleAvgIndex('.') 
    state_dict = torch.load(state_dir)
    _index.load_state_dict(state_dict)

    index = ModelAvgIndex(
            get_model(args, device),              # NOTE only supported with solo mode now.
            _index,
            include_buffers=True,
        )

    # compute all accuracies in advance and store
    accs = {}
    av_ckpts = list(state_dict['available_checkpoints'])
    av_ckpts.sort()

    # NOTE UNCOMMENT to precompute checkpoints
    # for ckpt in av_ckpts[:int(3*len(av_ckpts)//6)]:
    for ckpt in av_ckpts[:-1]:
        # model = index.avg_from(ckpt, until=av_ckpts[int(3*len(av_ckpts)//6)]) # until start of phase 2 (epoch 150)
        model = index.avg_from(ckpt, until=av_ckpts[-1])
        _, acc = eval_avg_model(model, train_loader, test_loader)
        accs[ckpt] = acc
        print(f'Step {ckpt}, acc: {acc}')
    torch.save(accs, os.path.join(save_dir, 'accs_computed.pt'))
    
    accs = torch.load(os.path.join(save_dir, 'accs_computed.pt'))
    # exponential_search(index, train_loader, test_loader, end=38400, start=38000, accs=accs, test=False)
    three_split_search(index, train_loader, test_loader, end=av_ckpts[-1], start=av_ckpts[0], accs=accs, test=False)

    # start = 43200
    # end = 58400
    # model = index.avg_from(start, until=end)
    # _, acc = eval_avg_model(model, train_loader, test_loader, compute_train_acc=True)   # NOTE there is some randomness in update_bn/evaluation? accuracies are ±0.2
    # print(acc)
    # update_bn(train_loader, model, device)
    # get_cifar_filtered_samples(args, ROOT_CLUSTER, model)

# python avg_index/search_avg.py --dataset=cifar100 --net=XX --expt_name=XX