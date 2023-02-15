from model.convnet import ConvNet, ConvNet_OP, MLP, LogisticRegression
from model.resnet import resnet20, resnet18, resnet50
from model.vgg import vgg16_C2
from model.vgg2 import vgg11, vgg11_bn, vgg16, vgg16_bn
from optimizer.optimizer import OptimizerEMA
import torch 

def get_model(args, device):
    if args.net == 'convnet':
        model = ConvNet()
    elif args.net == 'convnet_rgb':
        model = ConvNet(in_channels=3)
    elif args.net == 'convnet_op':
        model = ConvNet_OP()
    elif args.net == 'mlp':
        model = MLP()
    elif args.net == 'log_reg':
        model = LogisticRegression()
    elif args.net == 'rn20':
        model = resnet20(args)
    elif args.net == 'rn18':
        model = resnet18(args)
    elif args.net == 'rn50':
        model = resnet50(args)
    elif args.net == 'vgg':     # modified VGG-16 (Keskar et al, 2017)
        model = vgg16_C2(args)
    elif args.net == 'vgg11':   # modified VGG-11 (Beyond Spectral Gap)
        model = vgg11(args)
    elif args.net == 'vgg11bn': # modified VGG-11 with BN 
        model = vgg11_bn(args)
    elif args.net == 'vgg16':   # modified VGG-16 (less hidden units in linear layers)
        model = vgg16(args)
    elif args.net == 'vgg16bn':   # modified VGG-16 (less hidden units in linear layers)
        model = vgg16_bn(args)
    else:
        raise Exception('model not supported')

    return model.to(device)

def get_ema_models(args, models, device, alpha=0.995, ema_init=None, ramp_up=True):
    ema_models = []
    ema_opts = []
    for model in models:
        ema_model = get_model(args, device)
        if ema_init is not None:
            ema_model.load_state_dict(ema_init.state_dict())
        for param in ema_model.parameters():
            param.detach_()
        ema_opt = OptimizerEMA(model, ema_model, alpha=alpha, ramp_up=ramp_up)
        ema_models.append(ema_model)
        ema_opts.append(ema_opt)

    return ema_models, ema_opts

def add_noise_to_models(models, std, device):
    for model in models:
        for param in model.parameters():
            noise = torch.randn(*param.size()) * std    # Gaussian noise N(0, std^2) with param's size
            with torch.no_grad():
                param.add_(noise.to(device))
