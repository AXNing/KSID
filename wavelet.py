import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

class get_Fre(nn.Module):
    def __init__(self):
        super(get_Fre, self).__init__()

    def forward(self, dp):
        # print(dp.shape)
        
        
        # 执行傅里叶变换
        dp_fft = torch.fft.rfft2(dp, norm='backward')
        dp_amp = torch.abs(dp_fft)
        dp_pha = torch.angle(dp_fft)

        # 截取频谱图的前64列，得到128x64大小
        dp_amp = dp_amp[:,:,:, :64]
        dp_pha = dp_pha[:,:,:, :64]

        # print(dp_amp.shape)  # 应该输出 torch.Size([128, 64])
        
        return dp_amp, dp_pha

class Inv_Fre(nn.Module):
    def __init__(self):
        super(Inv_Fre, self).__init__()

    def forward(self, dp_amp, dp_pha, original_size):
        # Reconstruct complex tensor
        dp_complex = dp_amp * torch.exp(1j * dp_pha)
        
        # Perform inverse Fourier transform
        dp_inv = torch.fft.irfft2(dp_complex, s=original_size, norm='backward')
        
        return dp_inv

def Normalize(x):
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
