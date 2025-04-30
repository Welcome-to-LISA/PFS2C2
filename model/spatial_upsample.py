import torch
from torch.nn import init
import torch.nn as nn
import numpy as np


def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (height * weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('Spatial_upsample initialize network with %s' % init_type)
    net.apply(init_func)



def init_net(net, device, init_type, init_gain, initializer):
    print('spatial_upsample')
    net.to(device)
    if initializer:
        init_weights(net, init_type, init_gain)
    else:
        print('Spatial_upsample with default initialize')
    return net

def Spatial_upsample(args, invpsf, init_type='mean_space', init_gain=0.02, initializer=False):
        net = matrix_dot_lr2hr(invpsf)
        net.to(args.device)
        print('isCal_PSF==No,PllllSF is known as a prior information')
        return net

class InversePSF(nn.Module):
    def __init__(self, scale):
        super(InversePSF, self).__init__()
        self.inv_psf = nn.ConvTranspose2d(1, 1, scale, scale, 0, bias=False) # in_channels, out_channels, kernel_size, stride, padding

    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.psf(x[:, i, :, :].view(batch, 1, height, weight)) for i in range(channel)], 1)


class matrix_dot_lr2hr(nn.Module):
    def __init__(self, inv_PSF):
        super(matrix_dot_lr2hr, self).__init__()

        self.register_buffer('inv_PSF', inv_PSF.float().clone().detach())
        self.inv_conv = nn.ConvTranspose2d(1, 1, self.inv_PSF.shape[0], self.inv_PSF.shape[0], 0, bias=False)
        self.inv_conv.weight.data[0, 0] = self.inv_PSF
        self.inv_conv.requires_grad_(True)

    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.inv_conv(x[:, i, :, :].view(batch, 1, height, weight)) for i in range(channel)], 1)