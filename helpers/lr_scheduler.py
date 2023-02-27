import torch
from torch.optim import lr_scheduler as lrs
import numpy as np
import pdb 

def get_lr_schedulers(args, n_samples, opt):
    ''' WARNING only to be used with fixed batch size and n_nodes (i.e., fixed number of steps per epoch'''

    gamma = 1 / args.lr_decay_factor
    steps_per_epoch = np.ceil(n_samples / (args.n_nodes[0] * args.batch_size[0]))
    warmup_steps = max(steps_per_epoch * args.lr_warmup_epochs, 1)                          # NOTE need at least steps >= 1 to not break LinearLR
    phases_steps_1 = [steps_per_epoch * phase for phase in args.lr_decay]                   # to use in SequentialLR
    phases_steps_2 = [steps_per_epoch * phase - warmup_steps for phase in args.lr_decay]    # to use in MultiStepLR
    
    if not args.lr_scheduler:
        return lrs.ConstantLR(opt, 1, total_iters=0)        # NOTE to not interfere with outside tuning of LR

    if args.lr_decay_as == 'step':
        # warmup + (linear decay + constant) x times
        if args.lr_linear_decay_epochs > 0:
            lr_factors = [gamma**i for i in range(len(phases_steps_1)+1)]
            warmup = lrs.LinearLR(opt, start_factor=1e-8, end_factor=lr_factors[0], total_iters=warmup_steps)
            schs_list = [warmup]
            for i in range(len(phases_steps_1)):
                linear = lrs.LinearLR(opt, start_factor=lr_factors[i], end_factor=lr_factors[i+1], total_iters=steps_per_epoch*args.lr_linear_decay_epochs)
                schs_list.append(linear)
            scheduler = lrs.SequentialLR(opt, schs_list, milestones=phases_steps_1)
        # warmup + step decay
        else:
            warmup = lrs.LinearLR(opt, start_factor=1e-8, end_factor=1, total_iters=warmup_steps)    
            multistep = lrs.MultiStepLR(opt, milestones=phases_steps_2, gamma=1/args.lr_decay_factor)   
            scheduler = lrs.SequentialLR(opt, [warmup, multistep], milestones=[warmup_steps])
    elif args.lr_decay_as == 'cosine':
        warmup = lrs.LinearLR(opt, start_factor=1e-8, end_factor=1, total_iters=warmup_steps)  
        cosine = lrs.CosineAnnealingLR(opt, steps_per_epoch * args.epochs - warmup_steps)
        scheduler = lrs.SequentialLR(opt, [warmup, cosine], milestones=[warmup_steps])
    elif args.lr_decay_as == 'linear':
        warmup = lrs.LinearLR(opt, start_factor=1e-8, end_factor=1, total_iters=warmup_steps) 
        linear = lrs.LinearLR(opt, start_factor=1, end_factor=args.swa_lr/args.lr, total_iters=steps_per_epoch*args.lr_linear_decay_epochs)
        scheduler = lrs.SequentialLR(opt, [warmup, linear], milestones=phases_steps_1)
    else:
        raise ValueError('LR decay type not supported')

    return scheduler


def linear_for_warmup():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.BatchNorm1d(3),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    linear_scheduler = lrs.LinearLR(opt, start_factor=1e-8, end_factor=1, total_iters=10)

    return linear_scheduler


def sequential_test():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.BatchNorm1d(3),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    linear_scheduler = lrs.LinearLR(opt, start_factor=0.1, end_factor=1, total_iters=10)    # warm-up for 10 steps
    multistep_scheduler = lrs.MultiStepLR(opt, milestones=[20, 40], gamma=0.5)              # will decay at step 30 and 50 (start at 10)
    scheduler = lrs.SequentialLR(opt, [linear_scheduler, multistep_scheduler], milestones=[10])

    return scheduler

def sequential_test_2():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.BatchNorm1d(3),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    linear_scheduler = lrs.LinearLR(opt, start_factor=0.1, end_factor=1, total_iters=10)    # warm-up for 10 steps
    linear_scheduler_2 = lrs.LinearLR(opt, start_factor=1, end_factor=0.2, total_iters=5)              # will decay at step 30 and 50 (start at 10)
    linear_scheduler_3 = lrs.LinearLR(opt, start_factor=0.2, end_factor=0.04, total_iters=5)              # will decay at step 30 and 50 (start at 10)
    scheduler = lrs.SequentialLR(opt, [linear_scheduler, linear_scheduler_2, linear_scheduler_3], milestones=[30, 40])

    return scheduler

def test_warmup_and_linear_decay():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.BatchNorm1d(3),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    schedulers = []
    gamma = 0.2

    steps_per_epoch = 10
    warmup_steps = steps_per_epoch * 1  # 0-10 steps warming up
    phases_steps = [steps_per_epoch * phase - warmup_steps for phase in [3, 5, 7]]  # 20-30, 40-50, 60-70 decaying linearly

    lr_factors = [gamma**i for i in range(len(phases_steps)+1)]
    warmup = lrs.LinearLR(opt, start_factor=1e-8, end_factor=lr_factors[0], total_iters=warmup_steps)
    schs_list = [warmup]
    for i in range(len(phases_steps)):
        linear = lrs.LinearLR(opt, start_factor=lr_factors[i], end_factor=lr_factors[i+1], total_iters=steps_per_epoch*1)
        schs_list.append(linear)
    scheduler = lrs.SequentialLR(opt, schs_list, milestones=phases_steps)
    return scheduler


if __name__ == '__main__':
    # scheduler = linear_for_warmup() 
    scheduler = sequential_test()
    # scheduler = sequential_test_2()
    # scheduler = test_warmup_and_linear_decay()

    print(scheduler.optimizer.param_groups[0]['lr'])
    for _ in range(100):
        scheduler.step()
        print(scheduler.optimizer.param_groups[0]['lr'])