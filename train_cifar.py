import numpy as np
import pdb
from data.data import get_data
from topology import get_gossip_matrix, diffuse, get_average_model, get_average_opt
import time
import torch
from model.model import add_noise_to_models, get_model, get_ema_models
import torch.nn.functional as F
from helpers.utils import save_experiment, get_expt_name, AccuracyTracker, save_checkpoint
from helpers.logger import Logger
from helpers.parser import parse_args
from helpers.optimizer import get_optimizer, bn_update
from helpers.consensus import compute_node_consensus, compute_weight_distance, get_gradient_norm, compute_weight_norm
from helpers.evaluate import eval_all_models, evaluate_model
import wandb
import os

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
    ''' All-reduce all models and optimizers, and use to initialize new nodes (all of them with same params and momentum)'''
    avg_model = get_average_model(args, device, models)
    new_models = [get_model(args, device) for _ in range(n_nodes_new)]
    for i in range(len(new_models)):
        new_models[i].load_state_dict(avg_model.state_dict())
    
    opt_sd = get_average_opt(opts)
    new_opts = [get_optimizer(args, model) for model in new_models]
    for i in range(len(new_opts)):
        new_opts[i].load_state_dict(opt_sd)

    return new_models, new_opts

def initialize_nodes_no_mom(args, models, n_nodes_new, device):
    ''' Do not average momentum. Start new optimizers '''
    avg_model = get_average_model(args, device, models)
    new_models = [get_model(args, device) for _ in range(n_nodes_new)]
    for i in range(len(new_models)):
        new_models[i].load_state_dict(avg_model.state_dict())
    
    new_opts = [get_optimizer(args, model) for model in new_models]

    return new_models, new_opts

def init_nodes_EMA(args, models, ema_models, device, ramp_up=False):
    ''' Initialize EMA models for new nodes from an All-Reduce average of previous EMA models'''
    ema_avg_model = get_average_model(args, device, ema_models)
    new_ema_models, new_ema_opts = get_ema_models(args, models, device, ema_init=ema_avg_model, ramp_up=ramp_up)
    return new_ema_models, new_ema_opts

def update_SWA(args, swa_model, models, device, n):
    avg_model = get_average_model(args, device, models)
    if swa_model is None:
        swa_model = get_model(args, device)
        swa_model.load_state_dict(avg_model.state_dict())
    else:
        # difference to original implementation: they update BN parameters with a pass over data, we use the parameters at each point
        for swa_param, avg_param in zip(swa_model.state_dict().values(), avg_model.state_dict().values()):
            if swa_param.dtype == torch.float32:
                swa_param.mul_(n/(n+1))
                swa_param.add_(avg_param /(n+1))  
    n += 1
    return swa_model, n

def compute_model_tracking_metrics(args, logger, models, step, epoch, device, model_init=None):
    # consensus distance
    L2_dist = compute_node_consensus(args, device, models)
    logger.log_consensus(step, epoch, L2_dist)
    
    # weight distance to init
    if model_init is not None:
        L2_dist_init = compute_weight_distance(models[0], model_init)
        logger.log_weight_distance(step, epoch, L2_dist_init)
    
    # weight L2 norm
    L2_norm = compute_weight_norm(models[0])
    logger.log_weight_norm(step, epoch, L2_norm)
    
    # gradient L2 norm
    grad_norm = get_gradient_norm(models[0])
    logger.log_grad_norm(step, epoch, grad_norm)

########################################################################################


def train(args, steps, wandb):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('Random seed: ', args.seed)

    # data
    train_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction)
    if args.data_split:
        train_loader_lengths = [len(t) for t in train_loader]
        train_loader_iter = [iter(t) for t in train_loader]
        n_samples = np.sum([len(tl.dataset) for tl in train_loader])
    else:
        n_samples = len(train_loader.dataset)

    # init nodes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = [get_model(args, device) for _ in range(args.n_nodes[0])]
    opts = [get_optimizer(args, model) for model in models]
    if args.same_init:
        for i in range(1, len(models)):
            models[i].load_state_dict(models[0].state_dict())
    init_model = get_model(args, device)
    init_model.load_state_dict(models[0].state_dict())
    ema_models, ema_opts = get_ema_models(args, models, device)
    if args.late_ema_epoch > 0:
        late_ema_models, late_ema_opts = get_ema_models(args, models, device)
        late_ema_active = False   # indicate when to init step_offset
    swa_model = None
    swa_model2 = None # average every step, not every epoch (Moving Average)

    # initialize variables
    comm_matrix = get_gossip_matrix(args, 0)
    # print(comm_matrix)

    logger = Logger(wandb)
    ts_total = time.time()
    ts_steps_eval = time.time()

    phase = 0
    batch_size = args.batch_size[0]
    total_phases = len(args.start_epoch_phases)
    lr_decay_phase = 0
    total_lr_phases = len(args.lr_decay)
    step = 0
    epoch = 0
    max_acc = AccuracyTracker()
    max_ema_acc = AccuracyTracker()
    max_late_ema_acc = AccuracyTracker()
    max_swa2 = AccuracyTracker()
    n_swa, n_swa2 = 0, 0
    epoch_swa = args.epoch_swa # epoch to start SWA averaging (default: 100)

    # TRAIN LOOP
    # for step in range(steps['total_steps']):
    while epoch < args.epochs:
        if args.data_split:
            for i, l in enumerate(train_loader_lengths):
                if step % l == 0:
                    train_loader_iter[i] = iter(train_loader[i])

        # lr warmup
        if step < steps['warmup_steps']:
            lr = args.lr[0] * (step+1) / steps['warmup_steps']
            for opt in opts:
                for g in opt.param_groups:
                    g['lr'] = lr

        # lr decay
        if lr_decay_phase < total_lr_phases and epoch > args.lr_decay[lr_decay_phase]:
            lr_decay_phase += 1
            for opt in opts:
                for g in opt.param_groups:
                    g['lr'] = g['lr']/10
            print('lr decayed to %.4f' % g['lr'])

        # drop weight decay
        if args.wd_drop > 0 and epoch > args.wd_drop:
            for opt in opts:
                for g in opt.param_groups:
                    g['weight_decay'] = 0     

        # drop momentum
        if args.momentum_drop > 0 and epoch > args.momentum_drop:
            for opt in opts:
                for g in opt.param_groups:
                    g['momentum'] = 0  
                    g['nesterov'] = False

        # advance to the next training phase
        if phase+1 < total_phases and epoch > args.start_epoch_phases[phase+1]:
            phase += 1
            comm_matrix = get_gossip_matrix(args, phase)
            
            if len(args.batch_size) > 1:
                train_loader, _ = get_data(args, args.batch_size[phase])
                if args.data_split:
                    train_loader_lengths = [len(train_loader[0])]
                    train_loader_iter[0] = iter(train_loader[0])
                batch_size = args.batch_size[phase]

            # init new nodes
            if args.n_nodes[phase] > args.n_nodes[phase-1]: # if n_nodes doesn't change, no need to re-init models
                if args.init_momentum:
                    models, opts = initialize_nodes(args, models, opts, args.n_nodes[phase], device) 
                else:
                    models, opts = initialize_nodes_no_mom(args, models, args.n_nodes[phase], device)

                ema_models, ema_opts = init_nodes_EMA(args, models, ema_models, device)
                if args.late_ema_epoch > 0:
                    late_ema_models, late_ema_opts = init_nodes_EMA(args, models, late_ema_models, device, ramp_up=(not late_ema_active))

            # optionally, update lr
            if len(args.lr) > 1:
                for opt in opts:
                    for g in opt.param_groups:
                        g['lr'] = args.lr[phase]

            print('[Epoch %d] Changing to phase %d. Nodes: %d. Topology: %s. Local steps: %s.' % (epoch, phase, args.n_nodes[phase], args.topology[phase], args.local_steps[phase]))

        if args.model_std > 0:
            add_noise_to_models(models, args.model_std, device)

        # local update for each worker
        train_loss = 0
        ts_step = time.time()
        for i in range(len(models)):
            if args.data_split:
                train_loss += worker_local_step(models[i], opts[i], train_loader_iter[i], device)
            else:
                train_loss += worker_local_step(models[i], opts[i], iter(train_loader), device)
            
            # EMA updates
            ema_opts[i].update()
            if args.late_ema_epoch > 0 and epoch > args.late_ema_epoch:
                if not late_ema_active:
                    late_ema_active = True
                late_ema_opts[i].update()

        step +=1
        epoch += args.n_nodes[phase] * batch_size / n_samples
        train_loss /= args.n_nodes[phase]
        logger.log_step(step, epoch, train_loss, ts_total, ts_step)
        
        # gossip
        diffuse(args, phase, comm_matrix, models, step)
        
        # SWA update
        if epoch > epoch_swa:
            epoch_swa += 1
            swa_model, n_swa = update_SWA(args, swa_model, models, device, n_swa)
            test_loss, acc = evaluate_model(swa_model, test_loader, device)
            logger.log_acc(step, epoch, acc*100, name='SWA Accuracy')

        # MA update (SWA but every step)
        if epoch > args.epoch_swa:
            swa_model2, n_swa2 = update_SWA(args, swa_model2, models, device, n_swa2)


        # evaluate 
        if step % args.steps_eval == 0 or epoch >= args.epochs:
            with torch.no_grad():
                ts_eval = time.time()
                
                # evaluate on average of EMA models
                ema_model = get_average_model(args, device, ema_models)
                _, ema_acc = evaluate_model(ema_model, test_loader, device)
                logger.log_acc(step, epoch, ema_acc*100, 'EMA Accuracy')
                max_ema_acc.update(ema_acc)
                # Late EMA
                if late_ema_active:
                    late_ema_model = get_average_model(args, device, late_ema_models)
                    _, late_ema_acc = evaluate_model(late_ema_model, test_loader, device)
                    logger.log_acc(step, epoch, late_ema_acc*100, 'Late EMA Accuracy') 
                    max_late_ema_acc.update(late_ema_acc)
                # Moving Average
                if epoch > args.epoch_swa:
                    _, swa2_acc = evaluate_model(swa_model2, test_loader, device)
                    logger.log_acc(step, epoch, swa2_acc*100, 'MA Accuracy') 
                    max_swa2.update(swa2_acc)


                # evaluate on averaged model
                if args.eval_on_average_model:
                    ts_eval = time.time()
                    model = get_average_model(args, device, models)
                    test_loss, acc = evaluate_model(model, test_loader, device)
                    logger.log_eval(step, epoch, float(acc*100), test_loss, ts_eval, ts_steps_eval)
                    print('Epoch %.3f (Step %d) -- Test accuracy: %.2f -- EMA accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                        (epoch, step, float(acc*100), float(ema_acc*100), test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))
                    
                # evaluate on all models
                else:
                    acc, test_loss, acc_workers, loss_workers, acc_avg, test_loss_avg = eval_all_models(args, models, test_loader, device)
                    logger.log_eval_per_node(step, epoch, acc, test_loss, acc_workers, loss_workers, acc_avg, test_loss_avg, ts_eval, ts_steps_eval)
                    print('Epoch %.3f (Step %d) -- Test accuracy: %.2f -- EMA accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                        (epoch, step, acc, float(ema_acc*100), test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))
                    acc = acc_avg

                max_acc.update(acc)
                ts_steps_eval = time.time()

        # log consensus distance, weight norm
        if step % args.tracking_interaval == 0:
            compute_model_tracking_metrics(args, logger, models, step, epoch, device)

        # save checkpoint
        if args.save_model and step % args.save_interval == 0:
            # if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            for i in range(len(models)):
                save_checkpoint({
                    'epoch': epoch,
                    'step': step,
                    'net': args.net,
                    'state_dict': models[i].state_dict(),
                    'ema_state_dict': ema_models[i].state_dict(),
                    'optimizer' : opts[i].state_dict(),
                }, filename=SAVE_DIR + 'checkpoint_m' + str(i) + '.pth.tar')

                if args.wandb:
                    model_artifact = wandb.Artifact('ckpt_m' + str(i), type='model')
                    model_artifact.add_file(filename=SAVE_DIR + 'checkpoint_m' + str(i) + '.pth.tar')
                    wandb.log_artifact(model_artifact)
            print('Checkpoint(s) saved!')

    logger.log_single_acc(max_acc.get(), log_as='Max Accuracy')
    logger.log_single_acc(max_ema_acc.get(), log_as='Max EMA Accuracy')
    logger.log_single_acc(max_late_ema_acc.get(), log_as='Max Late EMA Accuracy')
    logger.log_single_acc(max_swa2.get(), log_as='Max MA Accuracy')

    # Make a full pass over EMA and SWA models to update 
    if epoch > args.epoch_swa:
        bn_update(args, train_loader, swa_model, device)
        _, swa_acc = evaluate_model(swa_model, test_loader, device)
        logger.log_single_acc(swa_acc, log_as='SWA Acc (after BN)')
        print('SWA Final Accuracy: %.2f' % swa_acc)

    bn_update(args, train_loader, ema_model, device)
    _, ema_acc = evaluate_model(ema_model, test_loader, device)
    logger.log_single_acc(ema_acc, log_as='EMA Acc (after BN)')
    print('EMA Final Accuracy: %.2f' % ema_acc)

    bn_update(args, train_loader, swa_model2, device)
    _, swa2_acc = evaluate_model(swa_model2, test_loader, device)
    logger.log_single_acc(swa2_acc, log_as='MA Acc (after BN)')
    print('MA Final Accuracy: %.2f' % ema_acc)

if __name__ == '__main__':
    from helpers.parser import SCRATCH_DIR, SAVE_DIR
    args = parse_args()
    os.environ['WANDB_CACHE_DIR'] = SCRATCH_DIR # NOTE this should be a directory periodically deleted. Otherwise, delete manually

    if not args.expt_name:
        args.expt_name = get_expt_name(args)
    
    steps = {
        'warmup_steps': 50000 / (args.n_nodes[0] * args.batch_size[0]) * args.lr_warmup_epochs,    # NOTE using n_nodes[0] to compute warmup epochs. Assuming warmup occurs in the first phase
    }

    if args.wandb:
        wandb.init(name=args.expt_name, dir=args.save_dir, config=args, project=args.project, entity=args.entity)
        train(args, steps, wandb)
        wandb.finish()
    else:
        train(args, steps, None)

# python train_cifar.py --lr=3.2 --topology=ring --dataset=cifar100 --wandb=False --local_exec=True --eval_on_average_model=True
# python train_cifar.py --lr=3.2 --topology=fully_connected --dataset=cifar100 --wandb=False --local_exec=True --model_std=0.01
# python train_cifar.py --lr=3.2 --topology ring fully_connected --local_steps 0 0 --dataset=cifar100 --wandb=False --local_exec=True --n_nodes 8 16 --start_epoch_phases 0 1 --eval_on_average_model=True --steps_eval=20 --lr 3.2 1.6 --late_ema_epoch=1
# python train_cifar.py --lr=3.2 --topology=ring --dataset=cifar100 --eval_on_average_model=True --n_nodes=4 --save_model=True --save_interval=20
# python train_cifar.py --lr=3.2 --topology solo solo --local_steps 0 0 --dataset=cifar100 --wandb=False --local_exec=True --n_nodes 1 1 --batch_size 1024 2048 --start_epoch_phases 0 1 --steps_eval=40 --lr 3.2 1.6 --data_split=True

