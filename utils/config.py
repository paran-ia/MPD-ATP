cfg ={
    'step':8,
    'batch_size':128,
    'learning_rate':0.1,
    'epochs':200,
    'datasets':'Fashion-MNIST',#Fashion-MNIST,CIFAR10,CIFAR100  N-MNIST,CIFAR10-DVS,DVS-Gesture
    'init_v_th':1.5,
    'theta':0.5,
    'DT_train':True,
    'DT_val':True,
    'epsilon':1.,
    'alpha':0.5,
    #======================噪声
    'salt_and_pepper_noise':False,
    'salt_prob':0.05,
    'pepper_prob':0.05,
    'gaussian_noise':False,
    'mean':0.,
    'std':0.05,
    'level':2,
    #======================记录数据
    'record_acc':False,
    'acc_save_path':'./exp/cifar10/01-29/10/acc.npy',

    'record_k':False,
    'k_save_path':'./exp/cifar10/01-29/10/k.npy',

    'record_vth':False,
    'vth_save_path':'./exp/cifar10/01-29/10/vth.npy',

    'record_fr':False,
    'fr_save_path': './exp/cifar10/01-29/10/fr.npy',
    #======================
    'start_cal_fr':False,#无需手动指定
    'fr_epoch':[],#无需手动指定

}