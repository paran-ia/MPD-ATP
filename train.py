import numpy as np
import torchvision
from braincog.datasets import get_cifar10_data, get_fashion_data, get_dvsc10_data, get_dvsg_data, get_cifar100_data, \
    get_mnist_data

import torch

import time

from spikingjelly.datasets.n_mnist import NMNIST
from torch.utils import data

#from data_process.build_data import build_cifar
from models.cifar import cifar_model
from models.mnist import mnist_model
from models.dvsgesture import DVSGestureNet
from models.cifardvs import CIFAR10DVSNet
import torch.nn as nn
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

#from models.spiking_models import CIFAR_NET
#from modules.neuron import PLIF
#from models.dvsgesture import CIFARNet
from utils.config import cfg
from utils.noise import add_salt_and_pepper_noise_tensor, add_gaussian_noise_tensor
import torch.nn.functional as F
from spikingjelly.activation_based.model.parametric_lif_net import CIFAR10Net
#from modules.func import LIF
from spikingjelly.activation_based.functional import reset_net,set_step_mode
from modules.neuron import PLIF
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_class = 10
    if cfg['datasets']== 'CIFAR10':#Fashion-MNIST,CIFAR10,CIFAR100  N-MNIST,CIFAR10-DVS,DVS-Gesture
        #train_loader, eval_loader = build_cifar('CIFAR10','./data/cifar10',cfg['batch_size'],0)
        train_loader, eval_loader, _, _ = get_cifar10_data(batch_size=cfg['batch_size'], num_workers=0,
                                                      root='./data/cifar10')
        snn = cifar_model(num_classes=10).to(device)
        #snn = CIFAR_NET().to(device)
        #snn =  CIFAR10Net(spiking_neuron=PLIF).to(device)
        #set_step_mode(snn,'m')
    elif cfg['datasets']== 'CIFAR100':#Fashion-MNIST,CIFAR10,CIFAR100  N-MNIST,CIFAR10-DVS,DVS-Gesture
        train_loader, eval_loader, _, _ = get_cifar100_data(batch_size=cfg['batch_size'], num_workers=0,
                                                       root='./data/cifar100')
        snn = cifar_model(num_classes=100).to(device)
        num_class = 100
    elif cfg['datasets']== 'Fashion-MNIST':
        train_loader, eval_loader, _, _ = get_fashion_data(batch_size=cfg['batch_size'], num_workers=0,
                                                       root='./data/Fashion-MNIST')
        snn = mnist_model().to(device)
    elif cfg['datasets'] == 'N-MNIST':
        train_set = NMNIST(root='./data/N-MNIST', train=True, data_type='frame', frames_number=cfg['step'], split_by='number')
        eval_set = NMNIST(root='./data/N-MNIST', train=False, data_type='frame', frames_number=cfg['step'],
                             split_by='number')
        train_loader = data.DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
        eval_loader = data.DataLoader(eval_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
        snn = mnist_model(in_channels=2).to(device)
    elif cfg['datasets'] == 'CIFAR10-DVS':
        train_loader, eval_loader, _, _ = get_dvsc10_data(batch_size=cfg['batch_size'], num_workers=0,step=cfg['step'],
                         root='./data/CIFAR10-DVS')
        snn = CIFAR10DVSNet().to(device)
    elif cfg['datasets'] == 'DVS-Gesture':
        train_loader, eval_loader, _, _ = get_dvsg_data(batch_size=cfg['batch_size'], num_workers=0,step=cfg['step'],
                         root='./data/DVS-Gesture')
        snn = DVSGestureNet().to(device)
        num_class = 11

    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_mse = nn.MSELoss().to(device)


    optimizer = torch.optim.SGD(snn.parameters(), lr=cfg['learning_rate'], momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=cfg['epochs'])
    # optimizer = torch.optim.Adam(snn.parameters(), lr=1e-3, weight_decay=1e-6, betas=(0.9, 0.999), eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    best_acc = 0
    best_epoch = 0
    acc_list = []
    k_list = []
    vth_list = []
    fr_list = []
    for epoch in range(cfg['epochs']):

        start_time = time.time()
        snn.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.long().to(device)
            if i==0:
                cfg['label']=labels[0]
            else:
                cfg['label']=None
            images = images.to(device)
            outputs = snn(images)
            loss = criterion_ce(outputs, labels)
            #loss = criterion_mse(outputs, F.one_hot(labels, num_class).float())
            if (i + 1) % 50 == 0:
                print("Loss: ", loss)
            loss.backward()
            optimizer.step()


        scheduler.step()
        correct = 0
        total = 0
        acc = 0

        #收集k的信息
        k_epoch = []
        vth_epoch = []
        for m in snn.modules():
            if isinstance(m, PLIF) and m.pos!='':
                if cfg['record_k']:
                    temp_tensor = m.k.clone().detach()[cfg[m.pos]].flatten().cpu()
                     # 生成随机索引
                    random_indices = torch.randperm(len(temp_tensor))[:200]
                    # 根据索引选取值
                    selected_values = temp_tensor[random_indices]
                    k_epoch.append(selected_values)
                if cfg['record_vth']:
                    temp_tensor = m.v_threshold.clone().detach()[cfg[m.pos]].flatten().cpu()
                    # 生成随机索引
                    #random_indices = torch.randperm(len(temp_tensor))[:100]
                    # 根据索引选取值
                    selected_values = temp_tensor[random_indices]
                    vth_epoch.append(selected_values)

        if cfg['record_k']:
            k_list.append(k_epoch)
        if cfg['record_vth']:
            vth_list.append(vth_epoch)





        # start testing
        snn.eval()
        with (torch.no_grad()):
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                if cfg['record_fr'] and batch_idx==0 and epoch%10 ==0:
                    cfg['start_cal_fr'] = True
                    cfg['fr_epoch'] = []
                inputs = inputs.to(device)
                if cfg['salt_and_pepper_noise']:
                    inputs = add_salt_and_pepper_noise_tensor(inputs,cfg['salt_prob'],cfg['pepper_prob'])
                elif cfg['gaussian_noise']:
                    inputs = add_gaussian_noise_tensor(inputs,cfg['mean'],cfg['std'])
                targets = targets.long().to(device)
                outputs = snn(inputs)
                if cfg['start_cal_fr']:
                    fr_list.append(cfg['fr_epoch'])
                    cfg['start_cal_fr'] = False
                _, predicted = outputs.cpu().max(1)
                total += (targets.size(0))
                correct += (predicted.eq(targets.cpu()).sum().item())

        acc = 100 * correct / total
        if cfg['record_acc']:
            acc_list.append(acc)
        print(f'Test Accuracy of the model on the test images: {acc}')
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch + 1
        print(f'best_acc is: {best_acc}')
        print(f'best_iter: {best_epoch}')
        print(f'Iters: {epoch + 1}\n')
        if cfg['record_acc'] and (epoch+1)%10==0:
            np.save(cfg['acc_save_path'],np.array(acc_list))
        if cfg['record_k'] and (epoch+1)%1==0:
            np.save(cfg['k_save_path'], np.array(k_list))
        if cfg['record_vth'] and (epoch+1)%1==0:
            np.save(cfg['vth_save_path'],np.array(vth_list))
        if cfg['record_fr'] and epoch%10==0:
            np.save(cfg['fr_save_path'],np.array(fr_list))