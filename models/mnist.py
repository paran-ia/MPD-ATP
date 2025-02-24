import torch
from braincog.base.encoder import Encoder
from spikingjelly.activation_based.functional import set_step_mode
from torch import nn
from spikingjelly.activation_based.layer import Conv2d, BatchNorm2d, MaxPool2d, Linear, Flatten
from modules.neuron import PLIF
from utils.config import cfg

class mnist_model(nn.Module):
    def __init__(self,in_channels=1):
        super().__init__()
        self.encoder = Encoder(cfg['step'],'direct')

        self.conv1 = Conv2d(in_channels,128,3,1,1)
        self.bn1 = BatchNorm2d(128)
        self.node1 = PLIF(pos='node1')
        self.pool1 = MaxPool2d(2)

        self.conv2 = Conv2d(128, 128, 3, 1, 1)
        self.bn2 = BatchNorm2d(128)
        self.node2 = PLIF(pos='node2')
        self.pool2 = MaxPool2d(2)

        self.flat = Flatten()
        if in_channels==2:
            self.fc1 = Linear(8192, 2048)
        else:
            self.fc1 = Linear(6272,2048)
        self.node3 = PLIF(pos='node3')
        self.node4 = PLIF()
        self.fc2 = Linear(2048, 10)
        set_step_mode(self, 'm')
    def forward(self,x):
        if x.dim()!=5:
            x = self.encoder(x)
        else:
            x = x.transpose(0, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.node1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.node2(x)
        x = self.pool2(x)

        x = self.flat(x)
        x = self.fc1(x)
        x = self.node4(x)

        x = self.fc2(x)
        x = self.node3(x)

        return x.mean(0)

if __name__ == '__main__':
    data = torch.rand(64,1,28,28).cuda()
    model = mnist_model().cuda()
    r = model(data)
    print(r.shape)




