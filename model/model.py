from model.convnet import ConvNet, ConvNet_OP, MLP
from model.resnet import resnet18
from model.vgg import VGG
from model.vgg2 import vgg11, vgg11_bn

def get_model(args, device):
    if args.net == 'convnet':
        model = ConvNet()
    elif args.net == 'convnet_op':
        model = ConvNet_OP()
    elif args.net == 'mlp':
        model = MLP()
    elif args.net == 'resnet18':
        model = resnet18(args)
    elif args.net == 'vgg':     # modified VGG-16 (Keskar et al, 2017)
        model = VGG()
    elif args.net == 'vgg11':   # modified VGG-11 (Beyond Spectral Gap)
        model = vgg11()
    elif args.net == 'vgg11bn': # modified VGG-11 with BN 
        model = vgg11_bn()
    else:
        raise Exception('model not supported')

    return model.to(device)

