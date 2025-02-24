import numpy as np
import torch
from braincog.base.encoder import Encoder
from spikingjelly.activation_based.functional import set_step_mode
from torch import nn
from spikingjelly.activation_based.layer import Conv2d, BatchNorm2d, MaxPool2d, Linear, Flatten
from modules.neuron import PLIF
from utils.config import cfg

class cifar_model(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.encoder = Encoder(cfg['step'],'direct')
        self.conv1 = Conv2d(3,128,3,1,1)
        self.bn1 = BatchNorm2d(128)
        self.node1 = PLIF()
        self.conv2 = Conv2d(128, 128, 3, 1, 1)
        self.bn2 = BatchNorm2d(128)
        self.node2 = PLIF()
        self.pool1 = MaxPool2d(2)
        self.conv3 = Conv2d(128,256,3,1,1)
        self.bn3 = BatchNorm2d(256)
        self.node3 = PLIF()
        self.conv4 = Conv2d(256, 256, 3, 1, 1)
        self.bn4 = BatchNorm2d(256)
        self.node4 = PLIF()
        self.pool2 = MaxPool2d(2)
        self.conv5 = Conv2d(256,512,3,1,1)
        self.bn5 = BatchNorm2d(512)
        self.node5 = PLIF()
        self.conv6 = Conv2d(512, 512, 3, 1, 1)
        self.bn6 = BatchNorm2d(512)
        self.node6 = PLIF()
        self.flat = Flatten()
        self.fc1 = Linear(512*8*8,512)
        self.fc2 = Linear(512, num_classes)
        self.node7 = PLIF(pos='node7')
        set_step_mode(self, 'm')

    def forward(self,x):
        x = self.encoder(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.node1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.node2(x)

        x = self.pool1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.node3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.node4(x)

        x = self.pool2(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.node5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.node6(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.node7(x)
        return x.mean(0)

if __name__ == '__main__':
    data = torch.rand(64,3,32,32)
    model = cifar_model()
    r = model(data)
    print(r.shape)




