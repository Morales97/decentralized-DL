

import math
from torchsummary import summary
import torch.nn as nn
import torch.nn.init as init
import pdb

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, n_classes):
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, n_classes),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))   # original VGG has a humongous linear layer (116M params, making the total 134M). And actually the features are upsampled from 512 to 4096, I think just by concatenation, so it doesn't add much
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, n_classes),
        # )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(args):
    """VGG 11-layer model (configuration "A")"""
    if args.dataset == 'cifar10':
        return VGG(make_layers(cfg['A']), 10)
    if args.dataset == 'cifar100':
        return VGG(make_layers(cfg['A']), 100)

def vgg11_bn(args):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    if args.dataset == 'cifar10':
        return VGG(make_layers(cfg['A'], batch_norm=True), 10)
    if args.dataset == 'cifar100':
        return VGG(make_layers(cfg['A'], batch_norm=True), 100)
    


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16(args):
    """VGG 16-layer model (configuration "D")"""
    if args.dataset == 'cifar10':
        return VGG(make_layers(cfg['D']), 10)
    if args.dataset == 'cifar100':
        return VGG(make_layers(cfg['D']), 100)

def vgg16_bn(args):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    if args.dataset == 'cifar10':
        return VGG(make_layers(cfg['D'], batch_norm=True), 10)
    if args.dataset == 'cifar100':
        return VGG(make_layers(cfg['D'], batch_norm=True), 100)

def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

if __name__ == '__main__':
    # model = vgg11()
    model = VGG(make_layers(cfg['D']), 100)
    summary(model, (3, 32, 32))
    pdb.set_trace()


