from copy import deepcopy

import PIL
import torch
from spikingjelly.activation_based.functional import set_step_mode
from spikingjelly.activation_based.layer import Conv2d, BatchNorm2d, MaxPool2d, Linear, Flatten
from torch import nn

from modules.neuron import PLIF


class DVSGestureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(2, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.node1 = PLIF()
        self.pool1 = MaxPool2d(kernel_size=2)

        self.conv2 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.node2 = PLIF()
        self.pool2 = MaxPool2d(kernel_size=2)

        self.conv3 = Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn3 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.node3 = PLIF()
        self.pool3 = MaxPool2d(kernel_size=2)

        self.conv4 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn4 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.node4 = PLIF()
        self.pool4 = MaxPool2d(kernel_size=2)

        self.flat = Flatten()
        self.fc1 = Linear(2304,512)
        self.node5 = PLIF()
        self.fc2 = Linear(512, 11)
        set_step_mode(self, 'm')

    def forward(self, x: torch.Tensor):
        x = x.transpose(0,1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.node1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.node2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.node3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.node4(x)
        x = self.pool4(x)

        x = self.flat(x)
        x = self.fc1(x)
        x = self.node5(x)
        x = self.fc2(x)
        return x.mean(0)

if __name__ == '__main__':
    pass
