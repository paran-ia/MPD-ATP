import math

import torch
from spikingjelly.activation_based.surrogate import PiecewiseLeakyReLU, ATan
from torch import nn
from utils.config import cfg


class PLIF(nn.Module):
    def __init__(self, init_tau=2.0, surrogate_function=ATan(),pos=''):
        super().__init__()
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))
        self.surrogate_function = surrogate_function
        self.v_threshold = cfg['init_v_th']  # c,h,w
        self.v = 0.  # b,c,h,w
        self.k = 0.
        self.pos=pos
        # if cfg['train_k']:
        #     self.k = nn.Parameter(torch.as_tensor(cfg['k']))
        # else:
        #     self.k = cfg['k']

    def neuronal_charge(self, x):
        # x:b,c,h,w
        # self.v = self.v + (x - self.v) * self.w.sigmoid()
        self.v = self.v + x

    def soft_reset(self, spike):
        self.v = self.v - spike * self.v_threshold

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def forward(self, x):
        # x:[t,b,c,h,w]
        if x.dim() == 5:
            t, b, c, h, w = x.shape
            self.v = torch.zeros(b, c, h, w).cuda()
            v_last = torch.zeros(c, h, w).cuda()

        else:
            t, b, l = x.shape
            self.v = torch.zeros(b, l).cuda()
            v_last = torch.zeros(l).cuda()

        output = []
        for i in range(t):
            input = x[i]

            self.neuronal_charge(input)
            if cfg['DT_train'] or cfg['DT_val']:
                v_cur = self.v.clone().detach()[5]  # c,h,w
                v_dif = v_cur - v_last  # 电压差
                v_last = v_cur
            spike = self.neuronal_fire()  # b,c,h,w

            if cfg['DT_train'] or cfg['DT_val']:
                # max = v_cur.max()
                # min = v_cur.min()
                self.k = v_cur.clone()
                # 应用条件
                # self.k[v_cur < 0] = cfg['theta']
                # self.k[v_cur > cfg['init_v_th']] = -cfg['theta']

                self.k[v_cur < cfg['alpha'] * cfg['init_v_th']] = cfg['theta']
                self.k[v_cur > (1 + cfg['alpha']) * cfg['init_v_th']] = -cfg['theta']
                # 使用掩码处理 0 到 2 范围的情况
                mask = (v_cur >= cfg['alpha'] * cfg['init_v_th']) & (v_cur <= (1 + cfg['alpha']) * cfg['init_v_th'])
                # self.k[mask] = cfg['init_v_th']/2. - v_cur[mask]
                self.k[mask] = -cfg['theta'] / (cfg['alpha'] * cfg['init_v_th']) * v_cur[mask] + cfg['theta'] / cfg[
                    'alpha']
                # self.k = 6.*v_cur / (max-min)-3.*(max+min)/(max-min)
                # self.k = 2.-torch.exp(torch.log(torch.as_tensor(3.))/3.*(v_cur-min))
                if self.training and cfg['DT_train']:
                    self.v_threshold = cfg['init_v_th'] - 0.5 * cfg['epsilon'] + cfg['epsilon'] / (
                                1 + torch.exp(self.k * v_dif))
                    self.v_threshold[v_dif <= 0] = cfg['init_v_th']
                    # self.v_threshold = 0.5*(cfg['init_v_th'] - 0.5*self.k*0.1 + self.k*0.1 / (1 + torch.exp(self.k * v_dif))+torch.where(v_cur < 1.2, torch.tensor(1.5), 1.5-0.5*torch.exp(-2*(v_cur-1.2))))
                    # self.v_threshold = cfg['init_v_th'] - 0.5*self.k*0.1 + self.k*0.1 / (1 + torch.exp(self.k * v_dif))
                    # self.v_threshold =torch.where(v_cur < 1.2, torch.tensor(1.5), 1.5-0.5*torch.exp(-2*(v_cur-1.2)))
                if not self.training and cfg['DT_val']:
                    self.v_threshold = cfg['init_v_th'] - 0.5 + 1 / (1 + torch.exp(self.k * v_dif))
                    self.v_threshold[v_dif <= 0] = cfg['init_v_th']
                    # self.v_threshold = 0.5*(cfg['init_v_th'] - 0.5*self.k*0.1 + self.k*0.1 / (1 + torch.exp(self.k * v_dif))+torch.where(v_cur < 1.2, torch.tensor(1.5), 1.5-0.5*torch.exp(-2*(v_cur-1.2))))
                    # self.v_threshold = cfg['init_v_th'] - 0.5*self.k*0.1 + self.k*0.1 / (1 + torch.exp(self.k * v_dif))
                    # self.v_threshold =torch.where(v_cur < 1.2, torch.tensor(1.5), 1.5-0.5*torch.exp(-2*(v_cur-1.2)))
            if cfg['label'] is not None and self.pos=='node3':
                print('*'*20)
                print(cfg['label'])
                print(self.v_threshold)
                print('*'*20)
            self.soft_reset(spike)
            output.append(spike)
        out = torch.stack(output)
        if cfg['start_cal_fr']:
            temp = out.clone().detach().cpu()
            fr = temp.mean()
            cfg['fr_epoch'].append(fr)
        return out


