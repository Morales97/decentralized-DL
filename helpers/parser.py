import argparse
import pdb

# Directory to save checkpoints to
# SAVE_DIR = '/mloraw1/danmoral/checkpoints/' 
SAVE_DIR = '/mlodata1/danmoral/checkpoints/' 

# wandb
SCRATCH_DIR = '/mloraw1/danmoral/scratch/' # directory to cache wandb artifacts in
ENTITY = 'morales97' # wandb username


def get_parser(parser=None):
    if not parser:
        parser = argparse.ArgumentParser(description='')

    # wandb 
    parser.add_argument('--expt_name', type=str, default='',
                        help='Name of the experiment for wandb')
    parser.add_argument('--wandb', type=boolfromstr, default=True,
                        help='whether or not to use wandb')  
    parser.add_argument('--local_exec', action='store_true',
                        help='local or cluster execution')  
    parser.add_argument('--project', type=str, default='MLO-CIFAR10',
                        help='wandb project to use')
    parser.add_argument('--entity', type=str, default=ENTITY,
                        help='wandb entity to use')

    # save
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR,
                        help='local directory to save experiment results to')
    parser.add_argument('--save_model', type=boolfromstr, default=True,
                        help='If not set, checkpoints will not be saved')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='step interval to save checkpoint') 
    parser.add_argument('--save_epoch_interval', type=int, default=1,
                        help='epoch interval to save checkpoint') 
    parser.add_argument('--save_final_model', type=boolfromstr, default=True,
                        help='save final model') 
    parser.add_argument('--save_best_model', type=boolfromstr, default=True,
                        help='save models with best accuracy') 
    parser.add_argument('--ema_acc_epoch', type=int, default=0,
                        help='epoch best EMA acc, to save checkpoint')      
    parser.add_argument('--ema_val_epoch', type=int, default=0,
                        help='epoch best EMA val, to save checkpoint')           

    # model
    parser.add_argument('--net', type=str, default='rn20',
                        help='choice of architecture') 
    parser.add_argument('--resume', type=str, default='',
                        help='path to checkpoint to resume from')
    parser.add_argument('--pretrained', type=str, default='',
                        help='path to checkpoint pretrained model')
    parser.add_argument('--freeze', action='store_true',
                        help='path to checkpoint pretrained model')

    # decentralized
    parser.add_argument('--n_nodes', type=int, nargs='+', default=[1],
                        help='number of nodes') 
    parser.add_argument('--same_init', type=boolfromstr, default=True,
                        help='initialize all models equally')
    parser.add_argument('-t', '--topology', type=str, nargs='+', default=['ring'],
                        help='topology of gossip matrix. see topology.py for options') 
    parser.add_argument('--local_steps', type=int, nargs='+', default=[0],
                        help='number of local steps inbetween each gossip') 
    parser.add_argument('--epochs', type=float, default=200,
                        help='number epochs to train for') 
    parser.add_argument('--start_epoch_phases', type=int, nargs='+', default=[0],
                        help='start epoch for each training phase. If [0], only one training phase') 
    parser.add_argument('--init_momentum', type=boolfromstr, default=True,
                        help='when increasing n_nodes, whether to init momentum')

    # model averaging
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.968, 0.984, 0.992, 0.996, 0.998],
                        help='EMA decay. Can specify many to keep multiple EMAs')   
    parser.add_argument('--ema_period', type=int, default=16,
                        help='period of steps to perform EMA update')                    
    parser.add_argument('--epoch_swa', type=int, default=100,
                        help='epoch when to start SWA averaging')
    parser.add_argument('--swa', action='store_true', 
                        help='Use SWA as in Izmailov et al.')
    parser.add_argument('--swa_lr', type=float, default=-1,
                        help='Final constant LR for SWA. If -1, do not use SWA scheduler')   
    parser.add_argument('--swa_per_phase', action='store_true', 
                        help='Compute SWA for each LR phase')
    parser.add_argument('--custom_a', type=float, default=0,
                        help='coefficient for custom SGD')  
    parser.add_argument('--custom_b', type=float, default=1,
                        help='coefficient for custom SGD')  
    parser.add_argument('--variant', type=int, default=0,
                        help='custom SGD variant to use')
    parser.add_argument('--avg_index', action='store_true', 
                        help='Save checkpoints of running model average')
    parser.add_argument('--log_train_ema', type=boolfromstr, default=True,
                        help='log EMA train accuracy and loss')

    # hyperparameters
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, nargs='+', default=[128],
                        help='batch size for a single node')
    parser.add_argument('--lr', type=float, nargs='+', default=[0.2],
                        help='learning rate')   
    parser.add_argument('--lr_decay', type=str, default='cosine',
                        help='type of lr decay (step/cosine/linear)')               
    # parser.add_argument('--lr_decay', type=int, nargs='+', default=[150, 225],
    #                     help='decay lr by factor at the listed fractions of training')                                       
    parser.add_argument('--lr_decay_factor', type=float, default=10,
                        help='lr decay factor') 
    parser.add_argument('--lr_warmup_epochs', type=int, default=5,
                        help='warm up learning rate in the first epochs') 
    parser.add_argument('--lr_linear_decay_epochs', type=int, default=0,
                        help='decay learning rate linearly insted of step') 
    parser.add_argument('--lr_scheduler', type=boolfromstr, default=True,
                        help='to use torch´s lr scheduler instead of manual')
    parser.add_argument('--final_lr', type=float, default=0,
                        help='final LR (usually to perform SWA in)') 

    # optimizer
    parser.add_argument('--opt', type=str, default='SGD',
                        help='optimizer for each node')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='')
    parser.add_argument('--nesterov', type=boolfromstr, default=True,
                        help='use nesterov momentum')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay, L2 regularization')

    # CR
    parser.add_argument('--cr_ema', type=float, default=None,
                        help='alpha for the EMA model used in Consistency Regularization')
    parser.add_argument('--lmbda', type=float, default=0.1,
                        help='weight for CR loss')

    # data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='') 
    parser.add_argument('--data_split', type=boolfromstr, default=True,
                        help='if False, sample with replacement from entire dataset (IID). if True, split dataset') 
    parser.add_argument('--p_label_skew', type=float, default=0,
                        help='Label skew for heterogeneity. Requires --data_split True.')    
    parser.add_argument('--data_fraction', type=float, default=-1,
                        help='Set between 0 and 1 to use a random subset of dataset with selected fraction of samples')
    parser.add_argument('--val_fraction', type=float, default=0.2,
                        help='Fraction for val/test split of test set')
    parser.add_argument('--viz_weights', action='store_true',
                        help='For Logistic regression, viz weights for each class')
    parser.add_argument('--label_noise', type=str, default=None,
                        help='Use noisy labels (from http://noisylabels.com/, 40% noise on C-100)')
    parser.add_argument('--select_samples', type=str, default='', 
                        help='File of saved subset to select. Used to train with noisy labels.')

    # evaluation
    parser.add_argument('--eval_on_average_model', type=boolfromstr, default=True,
                        help='evaluate on the average of all workers models (true), or separate and report mean accuracy (false)')
    parser.add_argument('--train_log_interval', type=int, default=100,
                        help='log train metrics every x number of steps')
    parser.add_argument('--steps_eval', type=int, default=400,
                        help='evaluate every x number of steps')
    parser.add_argument('--tracking_interval', type=int, default=200,
                        help='evaluate L2 distance of model conensus every x number of steps')
    parser.add_argument('--eval_after_epoch', action='store_true',
                        help='evaluate after each epoch or at step evaluation interval')
    parser.add_argument('--eval_on_test', type=boolfromstr, default=True,
                        help='evaluate best models on test set')
    parser.add_argument('--ema_bn_eval', action='store_true',
                        help='Evaluate on EMA with BN update')  
    return parser 

def check_assertions(args):
    if args.p_label_skew > 0:
        assert args.data_split == True, 'Sampling with replacement only available if split are not heterogeneous'

#def calculated_args(args):


def parse_args(parser=None):
    parser = get_parser(parser)
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