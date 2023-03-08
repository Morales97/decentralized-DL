from copy import deepcopy
import numpy as np
import pdb
from loaders.data import get_data
import time
import torch
from model.model import get_model, get_ema_model
import torch.nn.functional as F
from helpers.utils import TrainMetricsTracker, copy_checkpoint, get_folder_name, get_expt_name, MultiAccuracyTracker, save_checkpoint
from helpers.logger import Logger
from helpers.parser import parse_args
from optimizer.optimizer import get_optimizer
from helpers.consensus import compute_node_consensus, compute_weight_distance, get_momentum_norm, get_gradient_norm, compute_weight_norm
from helpers.train_dynamics import get_cosine_similarity, get_prediction_disagreement
from helpers.evaluate import eval_all_models, evaluate_model
from helpers.wa import AveragedModel, update_bn, SWALR
from avg_index.avg_index import TriangleAvgIndex, UniformAvgIndex, ModelAvgIndex
from helpers.lr_scheduler import get_lr_scheduler
import wandb
import os


def compute_model_tracking_metrics(args, logger, model, ema_models, opt, step, epoch, device, model_init=None):
    # weight distance to init
    if model_init is not None:
        L2_dist_init = compute_weight_distance(model, model_init)
        logger.log_weight_distance(step, epoch, L2_dist_init)
    
    # weight L2 norm
    L2_norm = compute_weight_norm(model)
    logger.log_weight_norm(step, epoch, L2_norm)
    
    # gradient L2 norm
    grad_norm = get_gradient_norm(model)
    logger.log_grad_norm(step, epoch, grad_norm)

    # EMA weight L2 norm
    ema_model = ema_models[args.alpha[-1]]  # norm of EMA model of node[0] with alpha[-1] 
    L2_norm_ema = compute_weight_norm(ema_model)
    logger.log_quantity(step, epoch, L2_norm_ema, name='EMA Weight L2 norm')

    # student to EMA weight L2 distance
    L2_dist_ema = compute_weight_distance(model, ema_model)
    logger.log_quantity(step, epoch, L2_dist_ema, name='Student-EMA L2 distance')

    # Momentum L2 norm
    if args.momentum > 0:
        mom_norm = get_momentum_norm(opt)
        logger.log_quantity(step, epoch, mom_norm , 'Momentum norm')

    # Cosine similarity Student-EMA
    cos_sim = get_cosine_similarity(model, ema_model)
    logger.log_quantity(step, epoch, cos_sim, name='Cosine similarity Student-EMA')
    
    # Cosine similarity with init
    if model_init is not None:
        cos_sim = get_cosine_similarity(model, model_init)
        logger.log_quantity(step, epoch, cos_sim, name='Cosine similarity to init')

    lr = opt.param_groups[0]['lr']
    logger.log_quantity(step, epoch, lr, name='Learing rate')

def update_bn_and_eval(model, train_loader, test_loader, device, logger, step=0, epoch=0, log_name=''):
    _model = deepcopy(model)
    update_bn(args, train_loader, _model, device)
    _, acc = evaluate_model(_model, test_loader, device)
    logger.log_quantity(step, epoch, acc, name=log_name)

########################################################################################


def train(args, wandb):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('Random seed: ', args.seed)

    # data
    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, args.val_fraction)
    train_loader = train_loader[0]
    n_samples = len(train_loader.dataset)

    # init model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(args, device)
    opt = get_optimizer(args, model)
    scheduler = get_lr_scheduler(args, n_samples, opt)

    # init averaged models
    ema_models, ema_opts = {}, {}
    for alpha in args.alpha:
        ema_models[alpha], ema_opts[alpha] = get_ema_model(args, model, device, alpha)
    swa_model = AveragedModel(model, device, use_buffers=True)

    # Average Index
    if args.avg_index:
        checkpoint_period = np.ceil(n_samples/args.batch_size[0]) * 2 // args.ema_period   # Saving index every 2 epochs -- empirically, this is more than often enough

        index = ModelAvgIndex(
            model,              
            UniformAvgIndex(get_folder_name(args), checkpoint_period=checkpoint_period),
            include_buffers=True,
        )
        # index = ModelAvgIndex(
        #     model,             
        #     TriangleAvgIndex(get_folder_name(args), checkpoint_period=checkpoint_period),
        #     include_buffers=True,
        # )

    # init variables
    logger = Logger(wandb)
    ts_total = time.time()
    ts_steps_eval = time.time()

    step = 0
    epoch = 0
    max_acc = MultiAccuracyTracker(['Student', 'EMA', *args.alpha])
    train_tracker = TrainMetricsTracker(['Student', *args.alpha])

    # Load checkpoint
    if args.resume:
        ckpt = torch.load(args.resume)
        
        model.load_state_dict(ckpt['state_dict'])
        ema_models[args.alpha[-1]].load_state_dict(ckpt['ema_state_dict'])  # TODO save and load other EMA
        opt.load_state_dict(ckpt['optimizer'])
        
        scheduler_state = ckpt['scheduler'] # NOTE if changing the LR scheduler (e.g., choosing a different final_lr), need to overwrite certain keys in the scheduler state_dict
        if 'prn164_SWA' in args.expt_name:
            scheduler_state['_schedulers'][1]['end_factor'] = args.final_lr / args.lr[0]   # NOTE change the end LR. Ad-hoc for SWA experiments

        scheduler.load_state_dict(scheduler_state)
        epoch = ckpt['epoch']
        step = ckpt['step']
        print(f'Resuming from step {step} (epoch {epoch}) ...')

    # TRAIN LOOP
    while epoch < args.epochs:
        
        for input, target in train_loader:
            ts_step = time.time()
            input = input.to(device)
            target = target.to(device)
            
            # Forward and backprop
            model.train()
            output = model(input)
            opt.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            opt.step()
            scheduler.step()    # NOTE could change scheduler to epoch-wise, which could simplify it (but complicate warmup...)
            
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            train_tracker.update('Student', correct, loss.item())

            # EMA updates
            if len(args.alpha) > 0 and step % args.ema_period == 0:
                for alpha in args.alpha:
                    ema_opts[alpha].update()
            
            # index model average
            if args.avg_index and step % args.ema_period == 0:
                index.record_step()

            step +=1
            epoch += input.shape[0] / n_samples
            logger.log_time_step(ts_step)

            if args.log_train_ema: 
                with torch.no_grad():
                    for alpha in args.alpha:
                        output = ema_models[alpha](input)
                        loss = F.cross_entropy(output, target)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct = pred.eq(target.view_as(pred)).sum().item() 
                        train_tracker.update(alpha, correct, loss.item())

            # Log train metrics
            if step % args.train_log_interval == 0:
                train_acc, train_loss = train_tracker.get('Student')
                logger.log_step(step, epoch, train_loss, train_acc, ts_total)
        
                # Log EMA train 
                if args.log_train_ema:
                    best_ema_acc, best_ema_loss = 0, 1e5
                    for alpha in args.alpha:
                        ema_acc, ema_loss = train_tracker.get(alpha)
                        logger.log_quantity(step, epoch, ema_acc, name=f'EMA {alpha} Train Acc')
                        logger.log_quantity(step, epoch, ema_loss, name=f'EMA {alpha} Train Loss')
                        best_ema_acc = max(best_ema_acc, ema_acc)
                        best_ema_loss = min(best_ema_loss, ema_loss)
                    logger.log_quantity(step, epoch, best_ema_acc, name=f'EMA Train Acc')
                    logger.log_quantity(step, epoch, best_ema_loss, name=f'EMA Train Loss')

            # log model tracking
            if step % args.tracking_interval == 0:
                compute_model_tracking_metrics(args, logger, model, ema_models, opt, step, epoch, device)

        # SWA update
        if epoch > args.epoch_swa:
            swa_model.update_parameters(model)

        # EVALUATE 
        with torch.no_grad():
            ts_eval = time.time()
            
            # evaluate on EMA models
            best_ema_acc = 0
            best_ema_loss = 1e5
            for alpha in args.alpha: 
                ema_loss, ema_acc = evaluate_model(ema_models[alpha], val_loader, device)
                logger.log_acc(step, epoch, ema_acc, ema_loss, name='EMA ' + str(alpha))
                max_acc.update(ema_acc, ema_loss, alpha)
                if ema_acc > best_ema_acc:
                    best_ema_acc = ema_acc
                    best_alpha = alpha
                best_ema_loss = min(best_ema_loss, ema_loss)
            max_acc.update(best_ema_acc, best_ema_loss, 'EMA')
            logger.log_acc(step, epoch, best_ema_acc, best_ema_loss, name='EMA')  
            
            # SWA
            if epoch > args.epoch_swa:
                update_bn_and_eval(swa_model, train_loader, val_loader, device, logger, step, epoch, log_name='SWA')

            # eval Student
            val_loss, acc = evaluate_model(model, val_loader, device)
            logger.log_eval(step, epoch, float(acc), val_loss, ts_eval, ts_steps_eval)
            print('Epoch %.3f (Step %d) -- Val accuracy: %.2f -- EMA accuracy: %.2f -- Val loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                (epoch, step, float(acc), float(ema_acc), val_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))
            max_acc.update(acc, val_loss, 'Student')
            ts_steps_eval = time.time()

        # save checkpoint periodically
        if args.save_model and np.round(epoch).astype(int) % args.save_epoch_interval == 0:
            save_checkpoint(args, model, ema_models, opt, scheduler, epoch, step, name='last', best_alpha=best_alpha)

            # save best checkpoints
            if max_acc.is_best_acc('Student'):
                copy_checkpoint(args, new_name='best_student_acc.pth.tar')
            if max_acc.is_best_acc('EMA'):
                copy_checkpoint(args, new_name='best_ema_acc.pth.tar')
            if max_acc.is_best_loss('Student'):
                copy_checkpoint(args, new_name='best_student_loss.pth.tar')
            if max_acc.is_best_loss('EMA'):
                copy_checkpoint(args, new_name='best_ema_loss.pth.tar')
        
    # save avg_index
    if args.avg_index:
        torch.save(index.state_dict(), os.path.join(get_folder_name(args), f'index_{index._index._uuid}_{step}.pt'))

    logger.log_single_acc(max_acc.get_acc('Student'), log_as='Max Val Accuracy')
    logger.log_single_acc(max_acc.get_acc('EMA'), log_as='Max EMA Val Accuracy')

    if epoch > args.epoch_swa:
        update_bn_and_eval(swa_model, train_loader, val_loader, device, logger, log_name='SWA Acc (after BN)')

    # eval best models on test set
    if args.eval_on_test:
        # student
        ckpt = torch.load(os.path.join(get_folder_name(args), 'best_student_acc.pth.tar'))
        model.load_state_dict(ckpt['state_dict'])
        epoch = ckpt['epoch']
        loss, acc = evaluate_model(model, test_loader, device)
        logger.log_single_acc(acc, log_as='Max Test Accuracy')
        print(f'Best student (not-averaged) model, from step {ckpt["step"]} (Epoch: {ckpt["epoch"]}) \tTest Accuracy: {acc} \tTest Loss: {loss}')
        
        # EMA
        ckpt = torch.load(os.path.join(get_folder_name(args), 'best_ema_acc.pth.tar'))
        alpha = ckpt['best_alpha']
        model.load_state_dict(ckpt['ema_state_dict_' + str(alpha)])
        epoch = ckpt['epoch']
        loss, acc = evaluate_model(model, test_loader, device)
        logger.log_single_acc(acc, log_as='Max EMA Test Accuracy')
        print(f'Best EMA model, alpha={alpha} at epoch {epoch} \tTest Accuracy: {acc} \tTest Loss: {loss}')


if __name__ == '__main__':
    args = parse_args()

    if not args.expt_name:
        args.expt_name = get_expt_name(args)
    if not os.path.exists(get_folder_name(args)):
        os.makedirs(get_folder_name(args))

    if args.wandb:
        wandb.init(name=args.expt_name, dir=args.save_dir, config=args, project=args.project, entity=args.entity)
        train(args, wandb)
        wandb.finish()
    else:
        train(args, None)

# python train_cifar.py --wandb=False --local_exec=True --net=rn18 --dataset=cifar100 --alpha 0.999 0.995 0.98 --avg_index
