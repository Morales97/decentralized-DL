import argparse
import pdb

# Directory to save checkpoints to
SAVE_DIR = '.' 

# wandb
SCRATCH_DIR = '/scratch/danmoral' # directory to cache wandb artifacts in
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
    parser.add_argument('--save_model', type=boolfromstr, default=True,
                        help='If not set, model will not be saved')

    # model
    parser.add_argument('--net', type=str, default='resnet18',
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

    # hyperparameters
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for a single node')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='learning rate')   
    parser.add_argument('--lr_decay', type=int, nargs='+', default=[150, 225],
                        help='decay lr by 10 at the listed fractions of training')                                       
    parser.add_argument('--lr_warmup_epochs', type=int, default=5,
                        help='warm up learning rate in the first epochs') 

    # optimizer
    parser.add_argument('--opt', type=str, default='SGD',
                        help='optimizer for each node')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='')
    parser.add_argument('--nesterov', type=boolfromstr, default=True,
                        help='use nesterov momentum')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay, L2 regularization')
    parser.add_argument('--gossip_momentum', type=boolfromstr, default=False,
                        help='if True, also communicate momentum (NOTE: to be implemented). otherwise, each model has its own momentum term')

    # data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='') 
    parser.add_argument('--data_split', type=boolfromstr, default=False,
                        help='if False, sample with replacement from entire dataset (IID). if True, split dataset') 
    parser.add_argument('--p_label_skew', type=float, default=0,
                        help='Label skew for heterogeneity. Requires data_split True.') 

    # evaluation
    parser.add_argument('--eval_on_average_model', type=boolfromstr, default=False,
                        help='evaluate on the average of all workers models (true), or separate and report mean accuracy (false)')
    parser.add_argument('--steps_eval', type=int, default=100,
                        help='evaluate every x number of steps')
    parser.add_argument('--steps_consensus', type=int, default=50,
                        help='evaluate L2 distance of model conensus every x number of steps')

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