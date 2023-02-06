from copy import deepcopy
import numpy as np
import pdb
from loaders.data import get_data
from topology import get_gossip_matrix, diffuse, get_average_model, get_average_opt
import time
import torch
import torch.optim as optim
from model.model import add_noise_to_models, get_model, get_ema_models
import torch.nn.functional as F
from helpers.utils import save_experiment, get_expt_name, MultiAccuracyTracker, save_checkpoint
from helpers.logger import Logger
from helpers.parser import parse_args
from helpers.optimizer import get_optimizer
from helpers.consensus import compute_node_consensus, compute_weight_distance, get_gradient_norm, compute_weight_norm
from helpers.evaluate import eval_all_models, evaluate_model
from helpers.wa import AveragedModel, update_bn, SWALR
from helpers.custom_sgd import CustomSGD
import wandb
import os


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
    model_x = get_model(args, device)
    model_y = get_model(args, device)
    model_v = get_model(args, device)
    model_y.load_state_dict(model_x.state_dict())
    model_v.load_state_dict(model_x.state_dict())
    if args.opt == 'SGD':
        opt = optim.SGD(model_y.parameters(), lr=args.lr[0], momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.wd)
    else:
        opt = CustomSGD(model_x.parameters(), model_y.parameters(), model_v.parameters(), \
            args.lr[0], alpha=args.alpha[0], beta=args.beta[0], variant=args.variant) # NOTE no momentum or weight decay

 

    logger = Logger(wandb)
    ts_total = time.time()
    ts_steps_eval = time.time()

    n_nodes = 1
    phase = 0
    batch_size = args.batch_size[0]
    total_phases = len(args.start_epoch_phases)
    lr_decay_phase = 0
    total_lr_phases = len(args.lr_decay)
    step = 0
    epoch = 0
    prev_epoch = 1
    max_acc = MultiAccuracyTracker(['X', 'Y', 'V'])
    if len(args.alpha) > 1:
        max_acc.init(args.alpha)
    epoch_swa = args.epoch_swa # epoch to start SWA averaging (default: 100)
    epoch_swa_budget = args.epoch_swa_budget

    # TRAIN LOOP
    # for step in range(steps['total_steps']):
    while epoch < args.epochs:
        if args.data_split:
            for i, l in enumerate(train_loader_lengths):
                if step % l == 0:
                    train_loader_iter[i] = iter(train_loader[i])

        # NOTE no warmup or decay until I figure out optimizers
        # lr warmup
        # if step < steps['warmup_steps']:
        #     lr = args.lr[0] * (step+1) / steps['warmup_steps']
        #     for opt in opts:
        #         for g in opt.param_groups:
        #             g['lr'] = lr
        # lr decay
        # if lr_decay_phase < total_lr_phases and epoch > args.lr_decay[lr_decay_phase]:
        #     lr_decay_phase += 1
        #     for opt in opts:
        #         for g in opt.param_groups:
        #             g['lr'] = g['lr']/args.lr_decay_factor
        #     print('lr decayed to %.4f' % g['lr'])


        # advance to the next training phase
        # if phase+1 < total_phases and epoch > args.start_epoch_phases[phase+1]:
        #     phase += 1
        #     comm_matrix = get_gossip_matrix(args, phase)
            
        #     if len(args.batch_size) > 1:
        #         print('batch_size updated to: ' + str(args.batch_size[phase]))
        #         train_loader, _ = get_data(args, args.batch_size[phase])
        #         if args.data_split:
        #             train_loader_lengths = [len(train_loader[0])]
        #             train_loader_iter[0] = iter(train_loader[0])
        #         batch_size = args.batch_size[phase]

        #     # init new nodes
        #     if len(args.n_nodes) > 1: 
        #         n_nodes = args.n_nodes[phase]
        #         print('n_nodes updated to: ' + str(n_nodes))
        #         if args.init_momentum:
        #             models, opts = initialize_nodes(args, models, opts, n_nodes, device) 
        #         else:
        #             models, opts = initialize_nodes_no_mom(args, models, n_nodes, device)

        #         ema_models, ema_opts = init_nodes_EMA(args, models, ema_models, device)  # does not support len(args.alpha) > 1
        #         if args.late_ema_epoch > 0:
        #             late_ema_models, late_ema_opts = init_nodes_EMA(args, models, late_ema_models, device, ramp_up=(not late_ema_active))

        #     # optionally, update lr
        #     if len(args.lr) > 1:
        #         print('New lr: ' + str(args.lr[phase]))
        #         for opt in opts:
        #             for g in opt.param_groups:
        #                 g['lr'] = args.lr[phase]

            # print('[Epoch %d] Changing to phase %d. Nodes: %d. Topology: %s. Local steps: %s.' % (epoch, phase, args.n_nodes[phase], args.topology[phase], args.local_steps[phase]))
            # print('[Epoch %d] Changing to phase %d.' % (epoch, phase))


        # local update for each worker
        train_loss = 0
        ts_step = time.time()

        input, target = next(train_loader_iter[0])
        input = input.to(device)
        target = target.to(device)

        model_y.train()
        output = model_y(input)
        if not args.opt == 'SGD':
            model_x.train() # running to keep BN statistis. Need to rethink this. Should BN stats be part of the optimization algo?
            _ = model_x(input)
            model_v.train()
            _ = model_v(input)
        opt.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        opt.step()

        step +=1
        epoch += n_nodes * batch_size / n_samples
        logger.log_step(step, epoch, loss.item(), ts_total, ts_step)
        
        
        # evaluate 
        if (not args.eval_after_epoch and step % args.steps_eval == 0) or epoch >= args.epochs or (args.eval_after_epoch and epoch > prev_epoch):
            prev_epoch += 1
            with torch.no_grad():
                ts_eval = time.time()
                
                # evaluate on average of EMA models
                # if len(args.alpha) == 1:
                #     ema_model = get_average_model(device, ema_models)
                #     ema_loss, ema_acc = evaluate_model(ema_model, test_loader, device)
                #     logger.log_acc(step, epoch, ema_acc*100, ema_loss, name='EMA')
                #     max_acc.update(ema_acc, 'EMA')
                # else:
                #     best_ema_acc = 0
                #     best_ema_loss = 10
                #     for alpha in args.alpha: 
                #         ema_model = get_average_model(device, ema_models[alpha])
                #         ema_loss, ema_acc = evaluate_model(ema_model, test_loader, device)
                #         logger.log_acc(step, epoch, ema_acc*100, ema_loss, name='EMA ' + str(alpha))
                #         max_acc.update(ema_acc, alpha)
                #         best_ema_acc = max(best_ema_acc, ema_acc)
                #         best_ema_loss = max(best_ema_loss, ema_loss)
                #     max_acc.update(best_ema_acc, 'EMA')
                #     logger.log_acc(step, epoch, best_ema_acc*100, best_ema_loss, name='EMA')  # actually EMA = multi-EMA. to not leave EMA empty
                #     logger.log_acc(step, epoch, best_ema_acc*100, name='Multi-EMA Best')
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


                # # evaluate on averaged model
                # if args.eval_on_average_model:
                #     ts_eval = time.time()
                #     model = get_average_model(device, models)
                #     test_loss, acc = evaluate_model(model, test_loader, device)
                #     logger.log_eval(step, epoch, float(acc*100), test_loss, ts_eval, ts_steps_eval)
                #     print('Epoch %.3f (Step %d) -- Test accuracy: %.2f -- EMA accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                #         (epoch, step, float(acc*100), float(ema_acc*100), test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))
                    
                # # evaluate on all models
                # else:
                #     acc, test_loss, acc_workers, loss_workers, acc_avg, test_loss_avg = eval_all_models(args, models, test_loader, device)
                #     logger.log_eval_per_node(step, epoch, acc, test_loss, acc_workers, loss_workers, acc_avg, test_loss_avg, ts_eval, ts_steps_eval)
                #     print('Epoch %.3f (Step %d) -- Test accuracy: %.2f -- EMA accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                #         (epoch, step, acc, float(ema_acc*100), test_loss, train_loss, time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))
                #     acc = acc_avg

                # eval model X
                ts_eval = time.time()
                if args.opt == 'SGD':
                    test_loss, acc_y = evaluate_model(model_y, test_loader, device)
                    logger.log_eval(step, epoch, float(acc_y*100), test_loss, ts_eval, ts_steps_eval)
                    print('Epoch %.3f (Step %d) -- Y accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                        (epoch, step, float(acc_y*100), test_loss, loss.item(), time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))
                else:
                    test_loss, acc_x = evaluate_model(model_x, test_loader, device)
                    _, acc_y = evaluate_model(model_y, test_loader, device)
                    _, acc_v = evaluate_model(model_v, test_loader, device)
                    logger.log_eval(step, epoch, float(acc_x*100), test_loss, ts_eval, ts_steps_eval)
                    logger.log_acc(step, epoch, acc_x*100, name='X')
                    logger.log_acc(step, epoch, acc_y*100, name='Y')
                    logger.log_acc(step, epoch, acc_v*100, name='V')
                    print('Epoch %.3f (Step %d) -- X accuracy: %.2f -- Y accuracy: %.2f -- V accuracy: %.2f -- Test loss: %.3f -- Train loss: %.3f -- Time (total/last/eval): %.2f / %.2f / %.2f s' %
                        (epoch, step, float(acc_x*100), float(acc_y*100), float(acc_v*100), test_loss, loss.item(), time.time() - ts_total, time.time() - ts_steps_eval, time.time() - ts_eval))

                    max_acc.update(acc_x, 'X')
                    max_acc.update(acc_v, 'V')
                max_acc.update(acc_y, 'Y')
                ts_steps_eval = time.time()

        # log consensus distance, weight norm
        # if step % args.tracking_interaval == 0:
        #     compute_model_tracking_metrics(args, logger, models, step, epoch, device)

        # save checkpoint
        # if args.save_model and step % args.save_interval == 0:
        #     # if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        #     for i in range(len(models)):
        #         save_checkpoint({
        #             'epoch': epoch,
        #             'step': step,
        #             'net': args.net,
        #             'state_dict': models[i].state_dict(),
        #             'ema_state_dict': ema_models[i].state_dict(),
        #             'optimizer' : opts[i].state_dict(),
        #         }, filename=SAVE_DIR + 'checkpoint_m' + str(i) + '.pth.tar')

        #         if args.wandb:
        #             model_artifact = wandb.Artifact('ckpt_m' + str(i), type='model')
        #             model_artifact.add_file(filename=SAVE_DIR + 'checkpoint_m' + str(i) + '.pth.tar')
        #             wandb.log_artifact(model_artifact)
        #     print('Checkpoint(s) saved!')

    logger.log_single_acc(max_acc.get('X'), log_as='Max X Accuracy')
    logger.log_single_acc(max_acc.get('Y'), log_as='Max Y Accuracy')
    logger.log_single_acc(max_acc.get('V'), log_as='Max V Accuracy')
    # logger.log_single_acc(max_acc.get('MA'), log_as='Max MA Accuracy')

    # Make a full pass over EMA and SWA models to update 
    # if epoch > args.epoch_swa:
    #     update_bn_and_eval(swa_model, train_loader, test_loader, device, logger, log_name='SWA Acc (after BN)')
    # if len(args.alpha) == 1:
    #     update_bn_and_eval(ema_model, train_loader, test_loader, device, logger, log_name='EMA Acc (after BN)')
    # update_bn_and_eval(swa_model2, train_loader, test_loader, device, logger, log_name='MA Acc (after BN)')
    # update_bn_and_eval(get_average_model(device, models), train_loader, test_loader, device, logger, log_name='Student Acc (after BN)') # TODO check if cumulative moving average BN is better than using running average

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

# python train_with_ema.py --wandb=False --local_exec=True --lr=0.1 --data_split=True --alpha=0.9 --beta=0
# python train_with_ema.py --project=MLO-optimizer --expt_name=a0.9_b0 --lr=0.1 --data_split=True --alpha=0.9 --beta=0 --steps_eval=400 --epochs=100
# python train_with_ema.py --project=MLO-optimizer --expt_name=a0.9_b0.9 --lr=0.1 --data_split=True --alpha=0.9 --beta=0.9 --steps_eval=400 --epochs=100
# python train_cifar.py --project=MLO-optimizer --expt_name=SGD_momentum --lr=0.1 --data_split=True --momentum=0.9 --nesterov=False --weight_decay=0 --n_nodes=1 --topology=solo --steps_eval=400 --epochs=100
# python train_with_ema.py --project=MLO-optimizer --expt_name=a0.9_b0.9_v1 --lr=0.1 --data_split=True --alpha=0.9 --beta=0.9 --variant=1 --steps_eval=400 --epochs=100
# python train_with_ema.py --project=MLO-optimizer --expt_name=a0.01_b0.5 --lr=0.1 --data_split=True --alpha=0.01 --beta=0.5 --steps_eval=400 --epochs=100
