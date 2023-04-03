import torch.optim as optim
import torch
import pdb
import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from optimizer.sgd_grad_stats import SGD_GradStats

def get_optimizer(args, model):
    if args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr[0], momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.wd)
    elif args.opt == 'SGD_GradStats':
        optimizer = SGD_GradStats(model.parameters(), lr=args.lr[0], momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.wd)
    else:
        raise Exception('Optimizer not supported')
    
    return optimizer


class Old_OptimizerEMA(object):
    '''
    Pseudo-optimizer. Performs Exponential Moving Average of model parameters
    '''
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())


    def update(self, step):
        _alpha = min(self.alpha, (step + 1)/(step + 10)) # ramp up EMA
        one_minus_alpha = 1.0 - _alpha

        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                # Update Exponential Moving Average parameters
                ema_param.mul_(_alpha)
                ema_param.add_(param * one_minus_alpha)


class OptimizerEMA(object):
    '''
    EMA optimizer which can optionally apply EMA to BN statistics, with eman=True (see EMAN paper by Cai et al)
    '''
    def __init__(self, model, ema_model, alpha=0.999, eman=True, ramp_up=True):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.eman = eman
        self.step = 0
        self.ramp_up = ramp_up


    def update(self):
        if self.ramp_up:
            _alpha = min(self.alpha, (self.step + 1)/(self.step + 10)) 
        else:
            _alpha = self.alpha
        self.step += 1
        one_minus_alpha = 1.0 - _alpha

        # update learnable parameters
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.mul_(_alpha)
            ema_param.add_(param * one_minus_alpha)

        if self.eman:
            # update buffers (aka, non-learnable parameters). These are usually only BN stats
            for buffer, ema_buffer in zip(self.model.buffers(), self.ema_model.buffers()):
                if ema_buffer.dtype == torch.float32:      
                    ema_buffer.mul_(_alpha)
                    ema_buffer.add_(buffer * one_minus_alpha)


class OptimizerEMA_IN(object):
    '''
    EMA optimizer which can optionally apply EMA to BN statistics (see EMAN paper by Cai et al)
    With eman=True, it should be exactly the same as OptimizerEMA
    Don't store model, pass it as attribute
    '''
    def __init__(self, alpha=0.999, eman=True, ramp_up=True):
        self.alpha = alpha
        self.eman = eman
        self.step = 0
        self.ramp_up = ramp_up

    def update(self, model, ema_model):
        if self.ramp_up:
            _alpha = min(self.alpha, (self.step + 1)/(self.step + 10)) 
        else:
            _alpha = self.alpha        
        self.step += 1
        one_minus_alpha = 1.0 - _alpha

        # update learnable parameters
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.mul_(_alpha)
            ema_param.add_(param * one_minus_alpha)

        if self.eman:
            # update buffers (aka, non-learnable parameters). These are usually only BN stats
            for buffer, ema_buffer in zip(model.buffers(), ema_model.buffers()):
                if ema_buffer.dtype == torch.float32:          
                    ema_buffer.mul_(_alpha)
                    ema_buffer.add_(buffer * one_minus_alpha)

