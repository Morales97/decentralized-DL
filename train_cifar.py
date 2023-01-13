import numpy as np
import pdb
from data.data import get_data
from topology import get_gossip_matrix, diffuse, get_average_model
import time
import torch
from model.model import get_model
import torch.nn.functional as F
from helpers.utils import save_experiment, get_expt_name
from helpers.logger import Logger
from helpers.parser import parse_args
from helpers.optimizer import get_optimizer
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
    models = [get_model(args, device) for _ in range(args.n_nodes)]
    opts = [get_optimizer(args, model) for model in models]
    if args.same_init:
        for i in range(1, len(models)):
            models[i].load_state_dict(models[0].state_dict())

    # gossip matrix
    comm_matrix = get_gossip_matrix(args)
    # print(comm_matrix)

    logger = Logger(wandb)
    ts_total = time.time()
    ts_steps_eval = time.time()

    # TRAIN LOOP
    for step in range(steps['total_steps']):

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
        if step in steps['decay_steps']:
            for opt in opts:
                for g in opt.param_groups:
                    g['lr'] = g['lr']/10
            print('lr decayed to %.4f' % g['lr'])

        # local update for each worker
        train_loss = 0
        ts_step = time.time()
        for i in range(len(models)):
            if args.data_split:
                train_loss += worker_local_step(models[i], opts[i], train_loader_iter[i], device)
            else:
                train_loss += worker_local_step(models[i], opts[i], iter(train_loader), device)
        
        epoch = step / steps['steps_per_epoch']
        train_loss /= args.n_nodes
        logger.log_step(step, epoch, train_loss, ts_total, ts_step)

        # gossip
        diffuse(args, comm_matrix, models, step, epoch)
        
        # evaluate 
        if (step+1) % args.steps_eval == 0 or (step+1) == steps['total_steps']:
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

            ts_steps_eval = time.time()


if __name__ == '__main__':
    from helpers.parser import SCRATCH_DIR
    args = parse_args()
    os.environ['WANDB_CACHE_DIR'] = SCRATCH_DIR

    if not args.expt_name:
        args.expt_name = get_expt_name(args)
    
    steps_per_epoch =  50000 // (args.batch_size * args.n_nodes)    # 50000 is number of training samples in CIFAR-10
    steps = {
        'steps_per_epoch': steps_per_epoch, 
        'total_steps': steps_per_epoch * args.epochs,
        'warmup_steps': steps_per_epoch * args.lr_warmup_epochs,
        'decay_steps': [np.floor(steps_per_epoch * args.epochs * decay) for decay in args.lr_decay]
    }

    if args.wandb:
        wandb.init(name=args.expt_name, dir=args.save_dir, config=args, project=args.project, entity=args.entity)
        train(args, steps, wandb)
        wandb.finish()
    else:
        train(args, steps, None)

