from model.convnet import ConvNet, ConvNet_OP, MLP
from model.resnet import resnet18
from model.vgg import VGG

def get_model(args, device):
    if args.net == 'convnet':
        model = ConvNet()
    elif args.net == 'convnet_op':
        model = ConvNet_OP()
    elif args.net == 'mlp':
        model = MLP()
    elif args.net == 'resnet18':
        model = resnet18(args)
    elif args.net == 'vgg':
        model = VGG(args)
    else:
        raise Exception('model not supported')

    return model.to(device)

