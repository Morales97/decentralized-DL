'''
Diferential Privacy
'''

from copy import deepcopy
import numpy as np
import pdb
from loaders.data import ROOT_CLUSTER, ROOT_LOCAL, get_data
from loaders.mnist import viz_weights, viz_weights_and_ema
from topology import get_gossip_matrix, diffuse, get_average_model, get_average_opt
import time
import torch
from model.model import add_noise_to_models, get_model, get_ema_models
import torch.nn.functional as F
from helpers.utils import save_experiment, get_expt_name, MultiAccuracyTracker, save_checkpoint
from helpers.logger import Logger
from helpers.parser import parse_args
from optimizer.optimizer import get_optimizer
from helpers.consensus import compute_node_consensus, compute_weight_distance, get_momentum_norm, get_gradient_norm, compute_weight_norm
from helpers.train_dynamics import get_cosine_similarity, get_prediction_disagreement
from helpers.evaluate import eval_all_models, evaluate_model
from helpers.wa import AveragedModel, update_bn, SWALR
from avg_index.avg_index import TriangleAvgIndex, UniformAvgIndex, ModelAvgIndex
from helpers.lr_scheduler import get_lr_schedulers
import wandb
import os

from torchvision import datasets, transforms
from torch.utils.data import TensorDataset


from pyvacy import optim, analysis, sampling

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
    if args.momentum > 0:
        mom_norm = get_momentum_norm(opts[0])
        logger.log_quantity(step, epoch, mom_norm , 'Momentum norm')

    # Cosine similarity Student-EMA
    cos_sim = get_cosine_similarity(models[0], ema_model)
    logger.log_quantity(step, epoch, cos_sim, name='Cosine similarity Student-EMA')
    
    # Cosine similarity with init
    if model_init is not None:
        cos_sim = get_cosine_similarity(models[0], model_init)
        logger.log_quantity(step, epoch, cos_sim, name='Cosine similarity to init')

    lr = opts[0].param_groups[0]['lr']
    logger.log_quantity(step, epoch, lr, name='Learing rate')

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

    # data - only C-10
    train_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction)
    if args.data_split:
        train_loader_lengths = [len(t) for t in train_loader]
        train_loader_iter = [iter(t) for t in train_loader]
        n_samples = np.sum([len(tl.dataset) for tl in train_loader])
    else:
        n_samples = len(train_loader.dataset)
        
    dataset = datasets.CIFAR10
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), 4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    traindata = dataset(
        root=ROOT_CLUSTER,
        train=True,
        transform=transform,
        download=True,
    )

    training_parameters = {
        'N': n_samples,
        # An upper bound on the L2 norm of each gradient update.
        # A good rule of thumb is to use the median of the L2 norms observed
        # throughout a non-private training loop.
        'l2_norm_clip': 1.0,
        # A coefficient used to scale the standard deviation of the noise applied to gradients.
        'noise_multiplier': 1.1,
        # Each example is given probability of being selected with minibatch_size / N.
        # Hence this value is only the expected size of each minibatch, not the actual. 
        'minibatch_size': 128,
        # Each minibatch is partitioned into distinct groups of this size.
        # The smaller this value, the less noise that needs to be applied to achieve
        # the same privacy, and likely faster convergence. Although this will increase the runtime.
        'microbatch_size': 1,
        # The usual privacy parameter for (ε,δ)-Differential Privacy.
        # A generic selection for this value is 1/(N^1.1), but it's very application dependent.
        'delta': 1e-5,
        # The number of minibatches to process in the training loop.
        'iterations': 390*200,
    }

    # init nodes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_nodes = args.n_nodes[0]
    models = [get_model(args, device) for _ in range(n_nodes)]
    # opts = [get_optimizer(args, model) for model in models]
    opts = [optim.DPSGD(params=models[0].parameters(), lr=args.lr[0], wd=args.wd, momentum=args.momentum, **training_parameters)]  # only support solo
    epsilon = analysis.epsilon(batch_size=training_parameters['minibatch_size'], **training_parameters)

    schedulers = [get_lr_schedulers(args, n_samples, opt) for opt in opts]   

    ema_models, ema_opts = {}, {}
    for alpha in args.alpha:
        ema_model_alpha, ema_opt_alpha = get_ema_models(args, models, device, alpha)
        ema_models[alpha] = ema_model_alpha
        ema_opts[alpha] = ema_opt_alpha

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
    epoch_swa = args.epoch_swa # epoch to start SWA averaging (default: 100)
    epoch_swa_budget = args.epoch_swa_budget
    epoch_swa3 = 0

    # Load checkpoint
    if args.resume:
        ckpt = torch.load(args.resume)
        assert len(models) == 1     # NOTE resuming only supported for solo training for now
        
        models[0].load_state_dict(ckpt['state_dict'])
        ema_models[args.alpha[-1]][0].load_state_dict(ckpt['ema_state_dict'])
        opts[0].load_state_dict(ckpt['optimizer'])
        
        scheduler_state = ckpt['scheduler'] # NOTE if changing the LR scheduler (e.g., choosing a different final_lr), need to overwrite certain keys in the scheduler state_dict
        if 'prn164_SWA' in args.expt_name:
            scheduler_state['_schedulers'][1]['end_factor'] = args.final_lr / args.lr[0]   # NOTE change the end LR. Ad-hoc for SWA experiments

        schedulers[0].load_state_dict(scheduler_state)
        epoch = ckpt['epoch']
        step = ckpt['step']
        print(f'Resuming from step {step} (epoch {epoch}) ...')

    # TRAIN LOOP
    # for step in range(steps['total_steps']):
    while epoch < args.epochs:


        train_loss = 0
        correct = 0
        ts_step = time.time()
        minibatch_loader, microbatch_loader = sampling.get_data_loaders(**training_parameters)
        for X_minibatch, y_minibatch in minibatch_loader(traindata):
            X_minibatch = X_minibatch.to(device)
            y_minibatch = y_minibatch.to(device)
            opts[0].zero_grad()
            for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
                opts[0].zero_microbatch_grad()
                loss = F.cross_entropy(models[0](X_microbatch), y_microbatch)
                loss.backward()
                opts[0].microbatch_step()
            opts[0].step()
        # for every node
        # for i in range(len(models)):
        #     if args.data_split:
        #         input, target = next(train_loader_iter[i])
        #     else:
        #         input, target = next(iter(train_loader))
        #     input = input.to(device)
        #     target = target.to(device)
        #     # Forward pass
        #     models[i].train()
        #     output = models[i](input)
        #     # Back-prop
        #     opts[i].zero_grad()
        #     loss = F.cross_entropy(output, target)
        #     loss.backward()
        #     opts[i].step()
        #     schedulers[i].step()
            
        #     train_loss += loss.item()
        #     pred = output.argmax(dim=1, keepdim=True)
        #     correct += pred.eq(target.view_as(pred)).sum().item()

            # EMA updates
            if len(args.alpha) > 0 and step % args.ema_interval == 0:
                for alpha in args.alpha:
                    ema_opts[alpha][0].update()
                # if args.late_ema_epoch > 0 and epoch > args.late_ema_epoch:
                #     if not late_ema_active:
                #         late_ema_active = True
                #     late_ema_opts[i].update()

        step +=1
        epoch += n_nodes * batch_size / n_samples
        train_loss /= n_nodes
        train_acc = correct / (n_nodes * batch_size) * 100
        logger.log_step(step, epoch, train_loss, train_acc, ts_total, ts_step)
        
        # EMA train log
        # if args.log_train_ema:
        #     ema_model = get_average_model(device, ema_models[args.alpha[-1]])   
        #     with torch.no_grad():
        #         output = ema_model(input)
        #         ema_loss = F.cross_entropy(output, target)
        #         pred = output.argmax(dim=1, keepdim=True)
        #         ema_acc = pred.eq(target.view_as(pred)).sum().item() / batch_size * 100
        #         logger.log_quantity(step, epoch, ema_loss.item(), name='EMA Train Loss')
        #         logger.log_quantity(step, epoch, ema_acc, name='EMA Train Acc')

        # gossip
        # diffuse(args, phase, comm_matrix, models, step)
        
        # SWA update
        # if epoch > epoch_swa:
        #     epoch_swa += 1
        #     swa_model.update_parameters(models)
        #     if args.swa_lr > 0:
        #         swa_scheduler.step()
        #     update_bn_and_eval(swa_model, test_loader, device, logger, log_name='SWA')
            
        #     if epoch > epoch_swa_budget:    # compute SWA at budget 1
        #         epoch_swa_budget = 1e5 # deactivate
        #         update_bn_and_eval(swa_model, train_loader, test_loader, device, logger, log_name='SWA Budget 1')

        # if args.swa_per_phase and epoch > epoch_swa3:   # TODO improve how to keep track of epoch end
        #     epoch_swa3 += 1
        #     swa_model3.update_parameters(models)


        # # MA update (SWA but every step)
        # if epoch > args.epoch_swa:
        #     swa_model2.update_parameters(models)

        # # index model average
        # if args.avg_index:
        #     index.record_step()

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
                # # Late EMA
                # if late_ema_active:
                #     late_ema_model = get_average_model(device, late_ema_models)
                #     late_ema_loss, late_ema_acc = evaluate_model(late_ema_model, test_loader, device)
                #     logger.log_acc(step, epoch, late_ema_acc*100, late_ema_loss, name='Late EMA') 
                #     max_acc.update(late_ema_acc, 'Late EMA')
                # # Moving Average
                # if epoch > args.epoch_swa:
                #     swa2_loss, swa2_acc = evaluate_model(swa_model2, test_loader, device)
                #     logger.log_acc(step, epoch, swa2_acc*100, swa2_loss, name='MA') 
                #     max_acc.update(swa2_acc, 'MA')


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
    #     if step % args.tracking_interval == 0:
    #         # get_prediction_disagreement(models[0], ema_models[args.alpha[-1]][0], test_loader, device)
    #         compute_model_tracking_metrics(args, logger, models, ema_models, opts, step, epoch, device, init_model)

    #     # save checkpoint
    #     if args.save_model and (step-1) % args.save_interval == 0:
    #         save_checkpoint(args, models, ema_models, opts, schedulers, epoch, step)

    # if args.save_final_model:
    #     save_checkpoint(args, models, ema_models, opts, schedulers, epoch, step)
        
    logger.log_single_acc(max_acc.get('Student'), log_as='Max Accuracy')
    logger.log_single_acc(max_acc.get('EMA'), log_as='Max EMA Accuracy')
    logger.log_single_acc(max_acc.get('Late EMA'), log_as='Max Late EMA Accuracy')
    # logger.log_single_acc(max_acc.get('MA'), log_as='Max MA Accuracy')

    # Make a full pass over EMA and SWA models to update 
    # if epoch > args.epoch_swa:
    #     update_bn_and_eval(swa_model, train_loader, test_loader, device, logger, log_name='SWA Acc (after BN)')
    # if args.swa_per_phase:
    #     lr_decay_phase += 1
    #     update_bn_and_eval(swa_model3, train_loader, test_loader, device, logger, log_name='SWA phase ' + str(lr_decay_phase))
    if len(args.alpha) == 1:
        update_bn_and_eval(ema_model, train_loader, test_loader, device, logger, log_name='EMA Acc (after BN)')
    update_bn_and_eval(swa_model2, train_loader, test_loader, device, logger, log_name='MA Acc (after BN)')
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
    if args.save_model and not os.path.exists(os.path.join(args.save_dir, args.dataset, args.net, args.expt_name)):
        os.makedirs(os.path.join(args.save_dir, args.dataset, args.net, args.expt_name))


    if args.wandb:
        wandb.init(name=args.expt_name, dir=args.save_dir, config=args, project=args.project, entity=args.entity)
        train(args, wandb)
        wandb.finish()
    else:
        train(args, None)

# python train_cifar_DP.py --dataset=cifar10 --wandb=False --alpha 0.999 0.995 --net=vgg16

