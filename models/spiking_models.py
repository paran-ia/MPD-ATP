import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.func import LIF,AdaptiveMaxPool2d
from spikingjelly.activation_based.layer import Conv2d,Dropout,Linear,AvgPool1d,BatchNorm2d
from spikingjelly.activation_based.functional import set_step_mode
class CIFAR_NET(nn.Module):
    def __init__(self, thresh=2.,tau=2., P=10, time_step=8):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn1 = BatchNorm2d(num_features=256)
        self.lif1 = LIF(thresh=thresh, tau=tau)
        self.conv2 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn2 = BatchNorm2d(num_features=256)
        self.lif2 = LIF(thresh=thresh, tau=tau)
        self.conv3 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn3 = BatchNorm2d(num_features=256)
        self.lif3 = LIF(thresh=thresh, tau=tau)
        self.pool1 = AdaptiveMaxPool2d(16)

        self.conv4 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn4 = BatchNorm2d(num_features=256)
        self.lif4 = LIF(thresh=thresh, tau=tau)
        self.conv5 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn5 = BatchNorm2d(num_features=256)
        self.lif5 = LIF(thresh=thresh, tau=tau)
        self.conv6 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn6 = BatchNorm2d(num_features=256)
        self.lif6 = LIF(thresh=thresh, tau=tau)
        self.pool2 = AdaptiveMaxPool2d(8)

        self.drop1 = Dropout(0.5)
        self.fc1 = Linear(8 * 8 * 256, 2048, bias=False)
        self.lif7 = LIF(thresh=thresh, tau=tau)

        self.drop2 = Dropout(0.5)
        self.fc2 = Linear(2048, 10*P, bias=False)
        self.lif8 = LIF(thresh=thresh, tau=tau)
        self.boost = nn.AvgPool1d(P, P)
        set_step_mode(self,'m')

    def forward(self,x):
        x = x.unsqueeze(0).repeat(8,1, 1, 1, 1)
        #x = (x >= torch.rand(x.size()).cuda()).float()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lif3(x)

        x= self.pool1(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lif4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.lif5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.lif6(x)

        x = self.pool2(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.lif7(x)

        x = self.drop2(x)
        x = self.fc2(x)
        x = self.lif8(x)

        x = self.boost(x).mean(0)
        return x
if __name__ == '__main__':
    data  =torch.rand(8,64,3,32,32)
    model = CIFAR_NET()
    r = model(data)
    print(r.shape)


