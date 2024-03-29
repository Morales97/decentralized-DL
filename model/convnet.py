
import torch.nn as nn
import torch
import pdb
from torchsummary import summary
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, in_channels=1):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool_adapt = nn.AdaptiveAvgPool2d((5, 5)) 
        self.fc1 = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.pool2(x)
        # x = torch.flatten(x, 1)
        x = self.pool_adapt(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


class ConvNet_OP(nn.Module):
    def __init__(self):
        super(ConvNet_OP, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(3200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self._init_zeros()
        
    def _init_zeros(self):
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)    # no activation function, softmax is embedded in CELoss
        output = F.log_softmax(x, dim=1)
        return output
    



if __name__ == '__main__':
    # model = MLP()
    model = ConvNet(in_channels=3)
    # model = ConvNet_OP()
    pdb.set_trace()
    summary(model, (1, 28, 28))
    pdb.set_trace()
