from model.convnet import ConvNet, ConvNet_OP, MLP
from model.resnet import resnet18

def get_model(config, device):
    if config['net'] == 'convnet':
        model = ConvNet()
    elif config['net'] == 'convnet_op':
        model = ConvNet_OP()
    elif config['net'] == 'mlp':
        model = MLP()
    elif config['net'] == 'resnet18':
        model = resnet18(config)
    else:
        raise Exception('model not supported')

    return model.to(device)

