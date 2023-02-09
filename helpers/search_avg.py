from readline import get_endidx
import torch
import torch.nn.functional as F
import numpy as np
import pdb

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


def eval_avg_model(start, end, train_loader, test_loader):
    '''
    Get model average between [start, end]
    Update BN stats on train data
    Evaluate on test data
    '''
    print('Evaluating average model from %d to %d...' % (start, end))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get average model
    model = None # TODO
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

def get_exp_idxs(start, min, ckpt_list):
    ''' get indices exponentially spaced'''
    start_idx = ckpt_list.index(start)
    min_idx = ckpt_list.index(min)
    idxs = []

    for i in range(int(np.log2(start_idx))+1):
        if start_idx - 2**i <= min_idx:  
            break
        idxs.append(start_idx - 2**i)

    if min_idx not in idxs:
        idxs.append(min_idx)
    return idxs

def exponential_search(train_loader, test_loader, end, start, min=0, accs={}, test=True):
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
        ckpt_list = []                              # TODO get ckpt_list
    else:
        ckpt_list = [1000*i for i in range(50)]
        score = lambda x: 1/((x-20)**2+1e-6)

    accs = accs
    assert start in ckpt_list, 'non-existing start checkpoint'
    assert end in ckpt_list, 'non-existing end checkpoint'

    # eval start model
    if not test:                                    # TODO remove after checking correct behavior
        _, acc = eval_avg_model(start, end, train_loader, test_loader)
    else:
        acc = score(ckpt_list.index(start))
        print('score %.8f at step %d' % (acc, start))
    accs[start] = acc
    best_acc = acc
    best_key = start
    
    # get indexes to search
    search_idxs = get_exp_idxs(start, min, ckpt_list)
    if test: print(search_idxs)
    
    for i, idx in enumerate(search_idxs):
        if ckpt_list[idx] not in accs.keys():
            if not test:                             # TODO remove after checking correct behavior
                _, acc = eval_avg_model(ckpt_list[idx], end, train_loader, test_loader)
            else:
                acc = score(idx)
                print('score %.8f at step %d' % (acc, ckpt_list[idx]))

            accs[ckpt_list[idx]] = acc
        if accs[ckpt_list[idx]] < best_acc:
            break
        else:
            best_acc = acc
            best_key = ckpt_list[idx]

    if best_key == start:
        return best_acc, start
    if best_key == ckpt_list[0]:
        return best_acc, ckpt_list[0]
    return exponential_search(end, ckpt_list[search_idxs[i-2]], min=ckpt_list[search_idxs[i]], accs=accs)