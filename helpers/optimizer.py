import torch.optim as optim
import torch
import pdb

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
        pdb.set_trace()


class NewOptimizerEMA(object):
    '''
    EMA optimizer which can optionally apply EMA to BN statistics (see EMAN paper by Cai et al)
    With eman=True, it should be exactly the same as OptimizerEMA
    '''
    def __init__(self, model, ema_model, alpha=0.999, eman=True):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.eman = eman


    def update(self, step):
        _alpha = min(self.alpha, (step + 1)/(step + 10)) # ramp up EMA
        one_minus_alpha = 1.0 - _alpha

        # update learnable parameters
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.mul_(_alpha)
            ema_param.add_(param * one_minus_alpha)

        if self.eman:
            # update buffers (aka, non-learnable parameters). These are usually only BN stats
            for buffer, ema_buffer in zip(self.model.buffers(), self.ema_model.buffers()):
                ema_buffer.mul_(_alpha)
                ema_buffer.add_(buffer * one_minus_alpha)


##### Update BN statistics (from SWA repo: https://github.com/timgaripov/swa/blob/411b2fcad59bec60c6c9eb1eb19ab906540e5ea2/utils.py#L74) #####

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, device):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.to(device)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)          # NOTE update momentum at every epoch to make an equal average, instead of moving
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))