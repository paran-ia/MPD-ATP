import torch


def add_salt_and_pepper_noise_tensor(tensor, salt_prob, pepper_prob):
    """
    给 PyTorch 张量添加椒盐噪声
    :param tensor: 输入图像张量，形状为 [batch_size, channels, height, width]
    :param salt_prob: 添加盐噪声的概率
    :param pepper_prob: 添加椒噪声的概率
    :return: 添加噪声后的图像张量
    """
    if tensor.dim()==4:
        B,C,H,W = tensor.shape
    else:
        B,T,C,H,W = tensor.shape
    noisy_tensor = tensor.clone()

    # 生成随机掩码

    if tensor.dim()==4:
        salt_mask = torch.rand_like(tensor[0][0]) < salt_prob
        pepper_mask = torch.rand_like(tensor[0][0]) < pepper_prob
        salt_mask = salt_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
        pepper_mask = pepper_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
    else:
        salt_mask = torch.rand_like(tensor[0][0][0]) < salt_prob
        pepper_mask = torch.rand_like(tensor[0][0][0]) < pepper_prob
        salt_mask = salt_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B,T, C, H, W)
        pepper_mask = pepper_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B,T, C, H, W)
    # 盐噪声（白色，像素值设置为1）
    noisy_tensor[salt_mask] = 1

    # 椒噪声（黑色，像素值设置为0）
    noisy_tensor[pepper_mask] = 0.0

    return noisy_tensor

def add_gaussian_noise_tensor(tensor, mean=0.0, std=0.1):
    """
    给 PyTorch 张量添加高斯噪声
    :param tensor: 输入图像张量，形状为 [batch_size, channels, height, width]
    :param mean: 高斯噪声的均值
    :param std: 高斯噪声的标准差
    :return: 添加噪声后的图像张量
    """
    if tensor.dim() == 4:
        B, C, H, W = tensor.shape
        noise = torch.randn_like(tensor[0][0]) * std + mean  # 生成高斯噪声
        noise = noise.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
    else:
        B, T, C, H, W = tensor.shape
        noise = torch.randn_like(tensor[0][0][0]) * std + mean  # 生成高斯噪声
        noise = noise.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B,T, C, H, W)


    noisy_tensor = tensor + noise*0.1  # 添加噪声
    noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)  # 限制值范围在 [0, 1]
    return noisy_tensor

