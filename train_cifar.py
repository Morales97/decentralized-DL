import numpy as np
import pdb
from data.data import get_data
from topology import get_gossip_matrix, diffuse, get_average_model, get_average_opt
import time
import torch
from model.model import get_model
import torch.nn.functional as F
from helpers.utils import save_experiment, get_expt_name
from helpers.logger import Logger
from helpers.parser import parse_args
from helpers.optimizer import get_optimizer
from helpers.consensus import compute_node_consensus, compute_weight_distance, get_gradient_norm
import wandb
import os

def evaluate_model(model, data_loader, device):
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # data = data.to(device)
            output = model(data)
            # output = model(data[None, ...])
            # sum up batch loss
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc

def eval_all_models(args, models, test_loader, device):
    acc_workers = []
    loss_workers = []
    for model in models:
        test_loss, acc = evaluate_model(model, test_loader, device)
        acc_workers.append(acc)
        loss_workers.append(test_loss)
    acc = float(np.array(acc_workers).mean()*100)
    test_loss = np.array(loss_workers).mean()

    model = get_average_model(args, device, models)
    test_loss_avg, acc_avg = evaluate_model(
        model, test_loader, device)
    
    return acc, test_loss, acc_workers, loss_workers, acc_avg, test_loss_avg

def worker_local_step(model, opt, train_loader_iter, device):
    input, target = next(train_loader_iter)
    input = input.to(device)
    target = target.to(device)

    model.train()
    output = model(input)
    opt.zero_grad()
    loss = F.cross_entropy(output, target)
    loss.backward()
    opt.step()

    return loss.item()


def initialize_nodes(args, models, opts, n_nodes_new, device):
    ''' All-reduce average all models and optimizers, and use to initialize new nodes (all of them with same params and momentum)'''
    avg_model = get_average_model(args, device, models)
    models = [get_model(args, device) for _ in range(n_nodes_new)]
    for i in range(len(models)):
        models[i].load_state_dict(avg_model.state_dict())
    
    opt_sd = get_average_opt(opts)
    opts = [get_optimizer(args, model) for model in models]
    for i in range(len(opts)):
        opts[i].load_state_dict(opt_sd)

    return models, opts

########################################################################################


def train(args, steps, wandb):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('Random seed: ', args.seed)

    # data
    train_loader, test_loader = get_data(args)
    if args.data_split:
        train_loader_lengths = [len(t) for t in train_loader]
        train_loader_iter = [iter(t) for t in train_loader]

    # init nodes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = [get_model(args, device) for _ in range(args.n_nodes[0])]
    opts = [get_optimizer(args, model) for model in models]
    if args.same_init:
        for i in range(1, len(models)):
            models[i].load_state_dict(models[0].state_dict())
    init_model = get_model(args, device)
    init_model.load_state_dict(models[0].state_dict())

    # gossip matrix
    comm_matrix = get_gossip_matrix(args, 0)
    # print(comm_matrix)

    logger = Logger(wandb)
    ts_total = time.time()
    ts_steps_eval = time.time()

    # TRAIN LOOP
    phase = 0
    total_phases = len(args.start_epoch_phases)
    lr_decay_phase = 0
    total_lr_phases = len(args.lr_decay)
    step = 0
    epoch = 0
    max_acc = 0
    # for step in range(steps['total_steps']):
    while epoch < args.epochs:
        if args.data_split:
            for i, l in enumerate(train_loader_lengths):
                if step % l == 0:
                    train_loader_iter[i] = iter(train_loader[i])

        # lr warmup
        if step < steps['warmup_steps']:
            lr = args.lr * (step+1) / steps['warmup_steps']
            for opt in opts:
                for g in opt.param_groups:
                    g['lr'] = lr

        # lr decay
        if lr_decay_phase < total_lr_phases and epoch > args.lr_decay[lr_decay_phase]:
            for opt in opts:
                for g in opt.param_groups:
                    g['lr'] = g['lr']/10
            print('lr decayed to %.4f' % g['lr'])

        # advance to the next training phase
        if phase+1 < total_phases and epoch > args.start_epoch_phases[phase+1]:
            phase += 1
            comm_matrix = get_gossip_matrix(args, phase)
            models, opts = initialize_nodes(args, models, opts, args.n_nodes[phase], device)
            print('[Epoch %d] Changing to phase %d. Nodes: %d. Topology: %s. Local steps: %s.' % (epoch, phase, args.n_nodes[phase], args.topology[phase], args.local_steps[phase]))

        # local update for each worker
        train_loss = 0
        ts_step = time.time()
        for i in range(len(models)):
            if args.data_split:
                train_loss += worker_local_step(models[i], opts[i], train_loader_iter[i], device)
            else:
                train_loss += worker_local_step(models[i], opts[i], iter(train_loader), device)
        
        step +=1
        epoch += args.n_nodes[phase] * args.batch_size / 50000
        train_loss /= args.n_nodes[phase]
        logger.log_step(step, epoch, train_loss, ts_total, ts_step)
        # gossip
        diffuse(args, phase, comm_matrix, models, step)
        
        # evaluate 
        if step % args.steps_eval == 0 or epoch >= args.epochs:
            ts_eval = time.time()
            
            # evaluate on averaged model
            if args.eval_on_average_model:
                ts_eval = time.time()
                model = get_average_model(args, device, models)
                test_loss, acc = evaluate_model(model, test_loader, device)
                logger.log_eval(step, epoch, float(acc*100), test_loss, ts_eval, ts_steps_eval)
                print('Epoch %.3f (Step %d) -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                      (epoch, step, float(acc*100), test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))
                
            # evaluate on all models
            else:
                acc, test_loss, acc_workers, loss_workers, acc_avg, test_loss_avg = eval_all_models(args, models, test_loader, device)
                logger.log_eval_per_node(step, epoch, acc, test_loss, acc_workers, loss_workers, acc_avg, test_loss_avg, ts_eval, ts_steps_eval)
                print('Epoch %.3f (Step %d) -- Test accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                      (epoch, step, acc, test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))

            if acc > max_acc:
                max_acc = acc

            ts_steps_eval = time.time()

        # evaluate consensus and L2 dist from init
        if step % args.steps_consensus == 0:
            L2_dist = compute_node_consensus(args, device, models)
            logger.log_consensus(step, epoch, L2_dist)
            L2_dist_init, L2_dist_l0 = compute_weight_distance(models[0], init_model)
            logger.log_weight_distance(step, epoch, L2_dist_init)
            logger.log_weight_distance_layer0(step, epoch, L2_dist_l0)
            grad_norm = get_gradient_norm(models[0])
            logger.log_grad_norm(step, epoch, grad_norm)

    logger.log_max_acc(max_acc)

if __name__ == '__main__':
    from helpers.parser import SCRATCH_DIR
    args = parse_args()
    os.environ['WANDB_CACHE_DIR'] = SCRATCH_DIR

    if not args.expt_name:
        args.expt_name = get_expt_name(args)
    
    steps = {
        'warmup_steps': 50000 / (args.n_nodes[0] * args.batch_size) * args.lr_warmup_epochs,    # NOTE using n_nodes[0] to compute warmup epochs. Assuming warmup occurs in the first phase
    }

    if args.wandb:
        wandb.init(name=args.expt_name, dir=args.save_dir, config=args, project=args.project, entity=args.entity)
        train(args, steps, wandb)
        wandb.finish()
    else:
        train(args, steps, None)

# python train_cifar_NEW.py --lr=3.2 --topology=ring --dataset=cifar100 --wandb=False --local_exec=True
# python train_cifar_NEW.py --lr=3.2 --topology ring fully_connected --local_steps 0 0 --dataset=cifar100 --wandb=False --local_exec=True --n_nodes 8 16 --start_epoch_phases 0 1 --eval_on_average_model=True