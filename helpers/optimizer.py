import torch.optim as optim
import torch

def get_optimizer(args, model):
    if args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr[0], momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.wd)
    else:
        raise Exception('Optimizer not supported')
    
    return optimizer


class OptimizerEMA(object):
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