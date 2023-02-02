import argparse
import pdb

# Directory to save checkpoints to
SAVE_DIR = '/mloraw1/danmoral/checkpoints/' 

# wandb
SCRATCH_DIR = '/mloraw1/danmoral/scratch/' # directory to cache wandb artifacts in
ENTITY = 'morales97' # wandb username


def get_parser():
    parser = argparse.ArgumentParser(description='')
    # wandb 
    parser.add_argument('--expt_name', type=str, default='',
                        help='Name of the experiment for wandb')
    parser.add_argument('--wandb', type=boolfromstr, default=True,
                        help='whether or not to use wandb')  
    parser.add_argument('--local_exec', type=boolfromstr, default=False,
                        help='local or cluster execution')  
    parser.add_argument('--project', type=str, default='MLO-CIFAR10',
                        help='wandb project to use')
    parser.add_argument('--entity', type=str, default=ENTITY,
                        help='wandb entity to use')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR,
                        help='local directory to save experiment results to')
    parser.add_argument('--save_model', type=boolfromstr, default=False,
                        help='If not set, model will not be saved')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='step interval to save checkpoint') 
                        
    # model
    parser.add_argument('--net', type=str, default='rn20',
                        help='choice of architecture') 
    
    # decentralized
    parser.add_argument('--n_nodes', type=int, nargs='+', default=[16],
                        help='number of nodes') 
    parser.add_argument('--same_init', type=boolfromstr, default=True,
                        help='initialize all models equally')
    parser.add_argument('-t', '--topology', type=str, nargs='+', default=['ring'],
                        help='topology of gossip matrix. see topology.py for options') 
    parser.add_argument('--local_steps', type=int, nargs='+', default=[0],
                        help='number of local steps inbetween each gossip') 
    parser.add_argument('--epochs', type=int, default=300,
                        help='number epochs to train for') 
    parser.add_argument('--start_epoch_phases', type=int, nargs='+', default=[0],
                        help='start epoch for each training phase. If [0], only one training phase') 
    parser.add_argument('--init_momentum', type=boolfromstr, default=True,
                        help='when increasing n_nodes, whether to init momentum')
    parser.add_argument('--epoch_swa', type=int, default=100,
                        help='epoch when to start SWA averaging')
    parser.add_argument('--late_ema_epoch', type=int, default=100,
                        help='epoch when to start Late EMA') 
    parser.add_argument('--model_std', type=float, default=0,
                        help='standard deviation for noise applied to each model`s parameters')  

    # hyperparameters
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, nargs='+', default=[128],
                        help='batch size for a single node')
    parser.add_argument('--lr', type=float, nargs='+', default=[0.2],
                        help='learning rate')   
    parser.add_argument('--lr_decay', type=int, nargs='+', default=[150, 225],
                        help='decay lr by factor at the listed fractions of training')                                       
    parser.add_argument('--lr_decay_factor', type=float, default=10,
                        help='lr decay factor') 
    parser.add_argument('--lr_warmup_epochs', type=int, default=5,
                        help='warm up learning rate in the first epochs') 
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.995],
                        help='EMA decaying rate')   

    # optimizer
    parser.add_argument('--opt', type=str, default='SGD',
                        help='optimizer for each node')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='')
    parser.add_argument('--nesterov', type=boolfromstr, default=True,
                        help='use nesterov momentum')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay, L2 regularization')
    parser.add_argument('--wd_drop', type=int, default=0,
                        help='[Experimental] Epoch where weight decay is dropped. If 0, do not drop') 
    parser.add_argument('--momentum_drop', type=int, default=0,
                        help='[Experimental] Epoch where momentum is dropped. If 0, do not drop') 

    # data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='') 
    parser.add_argument('--data_split', type=boolfromstr, default=False,
                        help='if False, sample with replacement from entire dataset (IID). if True, split dataset') 
    parser.add_argument('--p_label_skew', type=float, default=0,
                        help='Label skew for heterogeneity. Requires data_split True.')    
    parser.add_argument('--data_fraction', type=float, default=-1,
                        help='Set between 0 and 1 to use a random subset of dataset with selected fraction of samples')

    # evaluation
    parser.add_argument('--eval_on_average_model', type=boolfromstr, default=False,
                        help='evaluate on the average of all workers models (true), or separate and report mean accuracy (false)')
    parser.add_argument('--steps_eval', type=int, default=100,
                        help='evaluate every x number of steps')
    parser.add_argument('--tracking_interaval', type=int, default=50,
                        help='evaluate L2 distance of model conensus every x number of steps')
    parser.add_argument('--eval_after_epoch', action='store_true',
                        help='evaluate after each epoch or at step evaluation interval')
    return parser 

def check_assertions(args):
    if args.p_label_skew > 0:
        assert args.data_split == True, 'Sampling with replacement only available if split are not heterogeneous'

#def calculated_args(args):


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    check_assertions(args)
    return args

def get_default_args():
    parser = get_parser()
    args = parser.parse_args([])
    check_assertions(args)
    return args



def boolfromstr(s):
    if s.lower().startswith('true'):
        return True
    elif s.lower().startswith('false'):
        return False
    else:
        raise Exception('Incorrect option passed for a boolean')