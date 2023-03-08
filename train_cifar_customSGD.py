from copy import deepcopy
import numpy as np
import pdb
from loaders.data import get_data
from loaders.mnist import viz_weights, viz_weights_and_ema
from topology import get_gossip_matrix, diffuse, get_average_model, get_average_opt
import time
import torch
from model.model import get_model, get_ema_models
import torch.nn.functional as F
from helpers.utils import get_expt_name, MultiAccuracyTracker, save_checkpoint
from helpers.logger import Logger
from helpers.parser import parse_args
from optimizer.optimizer import get_optimizer
from helpers.consensus import compute_node_consensus, compute_weight_distance, get_momentum_norm, get_gradient_norm, compute_weight_norm
from helpers.train_dynamics import get_cosine_similarity, get_prediction_disagreement
from helpers.evaluate import eval_all_models, evaluate_model
from helpers.wa import AveragedModel, update_bn, SWALR
from avg_index.avg_index import UniformAvgIndex, ModelAvgIndex
from helpers.lr_scheduler import get_lr_scheduler
import wandb
from optimizer.custom_sgd import CustomSGD
import os

def initialize_nodes(args, models, opts, n_nodes_new, device):
    ''' All-reduce all models and optimizers, and use to initialize new nodes (all of them with same params and momentum)'''
    avg_model = get_average_model(device, models)
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
    avg_model = get_average_model(device, models)
    new_models = [get_model(args, device) for _ in range(n_nodes_new)]
    for i in range(len(new_models)):
        new_models[i].load_state_dict(avg_model.state_dict())
    
    new_opts = [get_optimizer(args, model) for model in new_models]

    return new_models, new_opts

def init_nodes_EMA(args, models, ema_models, device, ramp_up=False):
    ''' Initialize EMA models for new nodes from an All-Reduce average of previous EMA models'''
    ema_avg_model = get_average_model(device, ema_models)
    new_ema_models, new_ema_opts = get_ema_models(args, models, device, ema_init=ema_avg_model, ramp_up=ramp_up)
    return new_ema_models, new_ema_opts

def update_SWA(args, swa_model, models, device, n):
    avg_model = get_average_model(device, models)
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

def compute_model_tracking_metrics(args, logger, models, ema_models, opts, step, epoch, device, model_init=None):
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

    # EMA weight L2 norm
    ema_model = ema_models[args.alpha[-1]][0]  # norm of EMA model of node[0] with alpha[-1] 
    L2_norm_ema = compute_weight_norm(ema_model)
    logger.log_quantity(step, epoch, L2_norm_ema, name='EMA Weight L2 norm')

    # student to EMA weight L2 distance
    L2_dist_ema = compute_weight_distance(models[0], ema_model)
    logger.log_quantity(step, epoch, L2_dist_ema, name='Student-EMA L2 distance')

    # Momentum L2 norm
    if args.momentum > 0 and args.opt == 'SGD':
        mom_norm = get_momentum_norm(opts[0])
        logger.log_quantity(step, epoch, mom_norm , 'Momentum norm')

    # Cosine similarity Student-EMA
    cos_sim = get_cosine_similarity(models[0], ema_model)
    logger.log_quantity(step, epoch, cos_sim, name='Cosine similarity Student-EMA')
    
    # Cosine similarity with init
    if model_init is not None:
        cos_sim = get_cosine_similarity(models[0], model_init)
        logger.log_quantity(step, epoch, cos_sim, name='Cosine similarity to init')


def update_bn_and_eval(model, train_loader, test_loader, device, logger, log_name=''):
    _model = deepcopy(model)
    update_bn(args, train_loader, _model, device)
    _, acc = evaluate_model(_model, test_loader, device)
    logger.log_single_acc(acc, log_as=log_name)
    print(log_name + ' Accuracy: %.2f' % acc)

########################################################################################


def train(args, wandb):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('Random seed: ', args.seed)

    # data
    train_loader, _, test_loader = get_data(args, args.batch_size[0], args.data_fraction)
    if args.data_split:
        train_loader_lengths = [len(t) for t in train_loader]
        train_loader_iter = [iter(t) for t in train_loader]
        n_samples = np.sum([len(tl.dataset) for tl in train_loader])
    else:
        n_samples = len(train_loader.dataset)
    warmup_steps = n_samples / (args.n_nodes[0] * args.batch_size[0]) * args.lr_warmup_epochs

    # init nodes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_nodes = args.n_nodes[0]
    if args.opt != 'SGD':   # use Custom SGD
        models = [get_model(args, device)]  # model Y
        model_x = get_model(args, device)
        model_v = get_model(args, device)
        model_x.load_state_dict(models[0].state_dict())
        model_v.load_state_dict(models[0].state_dict())
        for param in model_x.parameters():
            param.detach_()
        for param in model_v.parameters():
            param.detach_()
        opts = [CustomSGD(model_x.parameters(), models[0].parameters(), model_v.parameters(), \
            args.lr[0], alpha=args.custom_a, beta=args.custom_b, variant=args.variant, weight_decay=args.wd)]
        if args.lr_scheduler:
            raise Exception('LR Scheduler does not work with custom optimizer. set --lr_scheduler=False')

    else:
        models = [get_model(args, device) for _ in range(n_nodes)]
        opts = [get_optimizer(args, model) for model in models]
        schedulers = get_lr_scheduler(args, n_samples, opts)   
    if args.same_init:
        for i in range(1, len(models)):
            models[i].load_state_dict(models[0].state_dict())
    init_model = get_model(args, device)
    init_model.load_state_dict(models[0].state_dict())

    # init model averaging
    ema_models, ema_opts = {}, {}
    for alpha in args.alpha:
        ema_model_alpha, ema_opt_alpha = get_ema_models(args, models, device, alpha)
        ema_models[alpha] = ema_model_alpha
        ema_opts[alpha] = ema_opt_alpha
    if args.late_ema_epoch > 0:
        late_ema_models, late_ema_opts = get_ema_models(args, models, device, alpha=0.995)
        late_ema_active = False   # indicate when to init step_offset

    if args.avg_index:
        index_save_dir = os.path.join(args.save_dir, args.expt_name)
        if not os.path.exists(index_save_dir):
            os.makedirs(index_save_dir)
        index = ModelAvgIndex(
            models[0],              # NOTE only supported with solo mode now.
            UniformAvgIndex(index_save_dir, checkpoint_period=args.steps_eval),
            include_buffers=True,
        )

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
    prev_epoch = 1
    max_acc = MultiAccuracyTracker(['Student', 'EMA', 'Late EMA', 'MA'])
    max_acc.init(args.alpha)

    # TRAIN LOOP
    # for step in range(steps['total_steps']):
    while epoch < args.epochs:
        if args.data_split:
            for i, l in enumerate(train_loader_lengths):
                if step % l == 0:
                    train_loader_iter[i] = iter(train_loader[i])

        # lr warmup
        if step < warmup_steps and not args.lr_scheduler:
            lr = args.lr[0] * (step+1) / warmup_steps
            for opt in opts:
                for g in opt.param_groups:
                    g['lr'] = lr

        # lr decay
        if lr_decay_phase < total_lr_phases and epoch > args.lr_decay[lr_decay_phase]:
            lr_decay_phase += 1
            if not args.lr_scheduler:
                for opt in opts:
                    for g in opt.param_groups:
                        g['lr'] = g['lr']/args.lr_decay_factor
                print('lr decayed to %.4f' % g['lr'])

        # advance to the next training phase
        if phase+1 < total_phases and epoch > args.start_epoch_phases[phase+1]:
            phase += 1
            comm_matrix = get_gossip_matrix(args, phase)
            
            if len(args.batch_size) > 1:
                print('batch_size updated to: ' + str(args.batch_size[phase]))
                train_loader, _, args, args.batch_size[phase])
                if args.data_split:
                    train_loader_lengths = [len(train_loader[0])]
                    train_loader_iter[0] = iter(train_loader[0])
                batch_size = args.batch_size[phase]

            # init new nodes
            if len(args.n_nodes) > 1: 
                n_nodes = args.n_nodes[phase]
                print('n_nodes updated to: ' + str(n_nodes))
                if args.init_momentum:
                    models, opts = initialize_nodes(args, models, opts, n_nodes, device) 
                else:
                    models, opts = initialize_nodes_no_mom(args, models, n_nodes, device)

                ema_models, ema_opts = init_nodes_EMA(args, models, ema_models, device)  # does not support len(args.alpha) > 1
                if args.late_ema_epoch > 0:
                    late_ema_models, late_ema_opts = init_nodes_EMA(args, models, late_ema_models, device, ramp_up=(not late_ema_active))

            # optionally, update lr
            if len(args.lr) > 1:
                print('New lr: ' + str(args.lr[phase]))
                for opt in opts:
                    for g in opt.param_groups:
                        g['lr'] = args.lr[phase]

            # print('[Epoch %d] Changing to phase %d. Nodes: %d. Topology: %s. Local steps: %s.' % (epoch, phase, args.n_nodes[phase], args.topology[phase], args.local_steps[phase]))
            print('[Epoch %d] Changing to phase %d.' % (epoch, phase))


        train_loss = 0
        correct = 0
        ts_step = time.time()
        # for every node
        for i in range(len(models)):
            if args.data_split:
                input, target = next(train_loader_iter[i])
            else:
                input, target = next(iter(train_loader))
            input = input.to(device)
            target = target.to(device)

            if args.opt != 'SGD':
                # ts_optimizer = time.time()
                models[i].train()
                output = models[i](input)
                opts[0].zero_grad()
                loss = F.cross_entropy(output, target)
                loss.backward()
                # ts_opt_step = time.time()
                opts[0].step()
                # opts[0].step_old()
                # print(f'Optimizer step time [s]: {time.time() - ts_opt_step}')

                
                with torch.no_grad():
                    model_x.train() # running to keep BN statistis. Need to rethink this. Should BN stats be part of the optimization algo?
                    _ = model_x(input)
                    if args.variant != 2:
                        model_v.train()
                        _ = model_v(input)

            else:
                # Forward pass
                models[i].train()
                output = models[i](input)
                # Back-prop
                opts[i].zero_grad()
                loss = F.cross_entropy(output, target)
                loss.backward()
                opts[i].step()
                schedulers[i].step()
                
            train_loss += loss.item()   # NOTE train loss for the last worker (no difference if n_nodes=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # EMA updates
            if len(args.alpha) > 0 and step % args.ema_period == 0:
                for alpha in args.alpha:
                    ema_opts[alpha][i].update()
                if args.late_ema_epoch > 0 and epoch > args.late_ema_epoch:
                    if not late_ema_active:
                        late_ema_active = True
                    late_ema_opts[i].update()

        step +=1
        epoch += n_nodes * batch_size / n_samples
        train_loss /= n_nodes
        train_acc = correct / (n_nodes * batch_size) * 100
        logger.log_step(step, epoch, train_loss, train_acc, ts_total, ts_step)
        
        # EMA train log
        if args.log_train_ema:
            ema_model = get_average_model(device, ema_models[args.alpha[-1]])   
            with torch.no_grad():
                output = ema_model(input)
                ema_loss = F.cross_entropy(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                ema_acc = pred.eq(target.view_as(pred)).sum().item() / batch_size * 100
                logger.log_quantity(step, epoch, ema_loss.item(), name='EMA Train Loss')
                logger.log_quantity(step, epoch, ema_acc, name='EMA Train Acc')

        # gossip
        diffuse(args, phase, comm_matrix, models, step)
        

        # index model average
        if args.avg_index:
            index.record_step()

        # evaluate 
        if (not args.eval_after_epoch and step % args.steps_eval == 0) or epoch >= args.epochs or (args.eval_after_epoch and epoch > prev_epoch):
            prev_epoch += 1
            with torch.no_grad():
                ts_eval = time.time()
                
                # evaluate on average of EMA models
                best_ema_acc = 0
                best_ema_loss = 1e5
                for alpha in args.alpha: 
                    ema_model = get_average_model(device, ema_models[alpha])
                    ema_loss, ema_acc = evaluate_model(ema_model, test_loader, device)
                    logger.log_acc(step, epoch, ema_acc*100, ema_loss, name='EMA ' + str(alpha))
                    max_acc.update(ema_acc, alpha)
                    best_ema_acc = max(best_ema_acc, ema_acc)
                    best_ema_loss = min(best_ema_loss, ema_loss)
                max_acc.update(best_ema_acc, 'EMA')
                logger.log_acc(step, epoch, best_ema_acc*100, best_ema_loss, name='EMA')  
                logger.log_acc(step, epoch, best_ema_acc*100, name='Multi-EMA Best')
                # Late EMA
                if late_ema_active:
                    late_ema_model = get_average_model(device, late_ema_models)
                    late_ema_loss, late_ema_acc = evaluate_model(late_ema_model, test_loader, device)
                    logger.log_acc(step, epoch, late_ema_acc*100, late_ema_loss, name='Late EMA') 
                    max_acc.update(late_ema_acc, 'Late EMA')


                if args.opt != 'SGD':   # custom SGD
                    acc_x, acc_y, acc_v = 0, 0, 0
                    test_loss, acc_x = evaluate_model(model_x, test_loader, device)
                    _, acc_y = evaluate_model(models[0], test_loader, device)
                    logger.log_eval(step, epoch, float(acc_y*100), test_loss, ts_eval, ts_steps_eval)
                    logger.log_acc(step, epoch, acc_x*100, name='X')
                    logger.log_acc(step, epoch, acc_y*100, name='Y')
                    if args.variant != 2:
                        _, acc_v = evaluate_model(model_v, test_loader, device)
                        logger.log_acc(step, epoch, acc_v*100, name='V')
                    print('Epoch %.3f (Step %d) -- X accuracy: %.2f -- Y accuracy: %.2f -- V accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                        (epoch, step, float(acc_x*100), float(acc_y*100), float(acc_v*100), test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))
                    acc=acc_y
                else:
                    # evaluate on averaged model
                    if args.eval_on_average_model:
                        ts_eval = time.time()
                        model = get_average_model(device, models)
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

                max_acc.update(acc, 'Student')
                ts_steps_eval = time.time()
                if args.viz_weights:
                    viz_weights_and_ema(models[0].linear.weight.detach().numpy(), ema_models[args.alpha[0]][0].linear.weight.detach().numpy(), save=True, epoch=epoch)

        # log consensus distance, weight norm
        if step % args.tracking_interval == 0:
            # get_prediction_disagreement(models[0], ema_models[args.alpha[-1]][0], test_loader, device)
            compute_model_tracking_metrics(args, logger, models, ema_models, opts, step, epoch, device, init_model)

        # save checkpoint
        if args.save_model and (step-1) % args.save_interval == 0:
            # if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            for i in range(len(models)):
                save_checkpoint({
                    'epoch': epoch,
                    'step': step,
                    'net': args.net,
                    'state_dict': models[i].state_dict(),
                    'ema_state_dict': ema_models[args.alpha[-1]][i].state_dict(),
                    'optimizer' : opts[i].state_dict(),
                }, filename=os.path.join(SAVE_DIR, args.expt_name, f'checkpoint_m{i}_{step}.pth.tar'))

                # if args.wandb:
                #     model_artifact = wandb.Artifact('ckpt_m' + str(i), type='model')
                #     model_artifact.add_file(filename=SAVE_DIR + 'checkpoint_m' + str(i) + '.pth.tar')
                #     wandb.log_artifact(model_artifact)
            print('Checkpoint(s) saved!')

    logger.log_single_acc(max_acc.get('Student'), log_as='Max Accuracy')
    logger.log_single_acc(max_acc.get('EMA'), log_as='Max EMA Accuracy')
    logger.log_single_acc(max_acc.get('Late EMA'), log_as='Max Late EMA Accuracy')
    # logger.log_single_acc(max_acc.get('MA'), log_as='Max MA Accuracy')

    # Make a full pass over EMA and SWA models to update 
    if len(args.alpha) == 1:
        update_bn_and_eval(ema_model, train_loader, test_loader, device, logger, log_name='EMA Acc (after BN)')
    update_bn_and_eval(get_average_model(device, models), train_loader, test_loader, device, logger, log_name='Student Acc (after BN)') # TODO check if cumulative moving average BN is better than using running average

    # save avg_index
    if args.avg_index:
        torch.save(index.state_dict(), os.path.join(index_save_dir, f'index_{index._index._uuid}_{step}.pt'))

    if args.viz_weights:
        # viz_weights(models[0].linear.weight.detach().numpy())
        # viz_weights(ema_models[args.alpha[0]][0].linear.weight.detach().numpy())
        viz_weights_and_ema(models[0].linear.weight.detach().numpy(), ema_models[args.alpha[0]][0].linear.weight.detach().numpy())

if __name__ == '__main__':
    from helpers.parser import SCRATCH_DIR, SAVE_DIR
    args = parse_args()
    #os.environ['WANDB_CACHE_DIR'] = SCRATCH_DIR # NOTE this should be a directory periodically deleted. Otherwise, delete manually

    if not args.expt_name:
        args.expt_name = get_expt_name(args)
    if args.save_model and not os.path.exists(os.path.join(SAVE_DIR, args.expt_name)):
        os.makedirs(os.path.join(SAVE_DIR, args.expt_name))


    if args.wandb:
        wandb.init(name=args.expt_name, dir=args.save_dir, config=args, project=args.project, entity=args.entity)
        train(args, wandb)
        wandb.finish()
    else:
        train(args, None)

# python train_cifar_customSGD.py --wandb=False --local_exec=True --opt=customSGD --variant=2 --custom_a=0.1 --custom_b=0.5 --epochs=50 --lr=0.1 --lr_warmup_epochs=0 --lr_decay=step --lr_decay=50 --net=rn20

# python train_cifar_customSGD.py --wandb=False --expt_name=new_a0_b1 --project=MLO-optimizer --opt=customSGD --momentum=0.9 --custom_a=0.1 --custom_b=0.5 --lr=0.1 --epochs=50 --lr_decay=100 --lr_warmup_epochs=0 --net=rn18
# python train_cifar_customSGD.py --wandb=False --expt_name=SGD --project=MLO-optimizer --momentum=0 --nesterov=False --wd=0 --lr=0.1 --epochs=50 --lr_decay=100 --lr_warmup_epochs=0 --net=rn18
# python train_cifar_customSGD.py --wandb=False --local_exec=True --expt_name=SGD --project=MLO-optimizer --momentum=0.9 --nesterov=True --wd=0 --lr=0.1 --epochs=50 --lr_decay=100 --lr_warmup_epochs=0 --net=rn18