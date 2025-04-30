# -*- coding: utf-8 -*-
"""

"""
import torch
import torch.nn as nn
from evalution import compute_sam, compute_psnr, compute_ergas, compute_cc, compute_rmse,compute_ssim
import numpy as np
import random
from options.visual import print_current_precision, print_options
import matplotlib.pyplot as plt
from options.visual import save_hhsi, save_wholehhsi

from model.spectral_upsample import Spectral_upsample
from model.spectral_downsample import Spectral_downsample
from model.spatial_downsample import Spatial_downsample
from model.spatial_upsample import Spatial_upsample

from options.config import args
from dataset.dataloader  import Dataset
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2)  # seed is set to 2

train_dataset = Dataset(args)
print("seddddfrergsrtgsdfgs")
hsi_channels = train_dataset.hsi_channel
msi_channels = train_dataset.msi_channel
sp_range = train_dataset.sp_range
sp_matrix = train_dataset.sp_matrix
psf = train_dataset.PSF
invpsf = train_dataset.InvPSF

# store the training configuration in opt.txt
print_options(args)

lhsi = train_dataset[0]["lhsi"].unsqueeze(0).to(args.device)
hmsi = train_dataset[0]['hmsi'].unsqueeze(0).to(args.device)
hhsi = train_dataset[0]['hhsi'].unsqueeze(0).to(args.device)
lrmsi_frommsi = train_dataset[0]['lrmsi_frommsi'].unsqueeze(0).to(args.device)
lrmsi_fromlrhsi = train_dataset[0]['lrmsi_fromlrhsi'].unsqueeze(0).to(args.device)

# reference 3-order H,W,C
hhsi_true = hhsi.detach().cpu().numpy()[0].transpose(1, 2, 0)
out_lrhsi_true = lhsi.detach().cpu().numpy()[0].transpose(1, 2, 0)
out_msi_true = hmsi.detach().cpu().numpy()[0].transpose(1, 2, 0)
out_frommsi_true = lrmsi_frommsi.detach().cpu().numpy()[0].transpose(1, 2, 0)
out_fromlrhsi_true = lrmsi_fromlrhsi.detach().cpu().numpy()[0].transpose(1, 2, 0)

# generate three modules
Spectral_up_net = Spectral_upsample(args, msi_channels, hsi_channels, init_type='normal', init_gain=0.02,initializer=False)
Spectral_down_net = Spectral_downsample(args, hsi_channels, msi_channels, sp_matrix, sp_range, init_type='Gaussian',init_gain=0.02, initializer=True)
Spatial_down_net = Spatial_downsample(args, psf, init_type='mean_space', init_gain=0.02, initializer=True)
Spatial_up_net = Spatial_upsample(args, invpsf, init_type='mean_space', init_gain=0.02, initializer=True)

optimizer_Spectral_down = torch.optim.Adam(Spectral_down_net.parameters(), lr=args.lr_stage1)
optimizer_Spatial_down = torch.optim.Adam(Spatial_down_net.parameters(), lr=args.lr_stage1)
optimizer_Spectral_up = torch.optim.Adam(Spectral_up_net.parameters(), lr=args.lr_stage2)
optimizer_Spatial_up = torch.optim.Adam(Spatial_up_net.parameters(), lr=args.lr_stage2)

L1Loss = nn.L1Loss(reduction='mean')
'''
#begin stage 1
'''
print('________________________Stage 1____________________________')
s1_start_time = time.time()


for epoch in range(1, args.epoch_stage1 + 1):
    lr = args.lr_stage1
    optimizer_Spatial_up.zero_grad()
    optimizer_Spectral_up.zero_grad()
    #up
    out_hrhsi_fromlrhsi = Spatial_up_net(lhsi)
    out_hrmsi_frommsi = Spectral_up_net(hmsi)
    loss0 = L1Loss(out_hrhsi_fromlrhsi, out_hrmsi_frommsi)
    loss0.backward()
    optimizer_Spatial_up.step()
    optimizer_Spectral_up.step()

    if epoch % 1000 == 0:
        with torch.no_grad():
            out_hrhsi = out_hrhsi_fromlrhsi.detach().cpu().numpy()[0].transpose(1, 2, 0)
            out_hrmsi = out_hrmsi_frommsi.detach().cpu().numpy()[0].transpose(1, 2, 0)
            train_message = 'S1-up, epoch:{} lr:{}\ntrain:L1loss:{}, sam_loss:{}, psnr:{}, CC:{}, rmse:{}'. \
                format(epoch, lr,
                       np.mean(np.abs(out_hrhsi - out_hrmsi)),
                       compute_sam(out_hrhsi, out_hrmsi),
                       compute_psnr(out_hrhsi, out_hrmsi),
                       compute_cc(out_hrhsi, out_hrmsi),
                       compute_rmse(out_hrhsi, out_hrmsi)
                       )
            print(train_message)
            print('\n')

    if (epoch > args.decay_begin_epoch_stage1 - 1):
        each_decay = args.lr_stage1 / (args.epoch_stage1 - args.decay_begin_epoch_stage1 + 1)
        lr = lr - each_decay
        for param_group in optimizer_Spatial_up.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_Spectral_up.param_groups:
            param_group['lr'] = lr

print_current_precision(args, 'Stage 1')
print_current_precision(args, train_message)

'''
#begin stage 2
'''
print('________________________Stage2____________________________')


for epoch in range(1, args.epoch_stage2 + 1):
    lr = args.lr_stage2
    optimizer_Spatial_down.zero_grad()
    optimizer_Spectral_down.zero_grad()
    #down
    out_lrmsi_fromlrhsi = Spectral_down_net(lhsi)
    out_lrmsi_frommsi = Spatial_down_net(hmsi)
    loss1 = L1Loss(out_lrmsi_fromlrhsi, out_lrmsi_frommsi)
    loss1.backward()
    optimizer_Spatial_down.step()
    optimizer_Spectral_down.step()

    if epoch % 1000 == 0:
        with torch.no_grad():
            out_lrhsi = out_lrmsi_fromlrhsi.detach().cpu().numpy()[0].transpose(1, 2, 0)
            out_lrmsi = out_lrmsi_frommsi.detach().cpu().numpy()[0].transpose(1, 2, 0)

            print('estimated PSF:', Spatial_down_net.psf.weight.data)
            print('true PSF:', psf)
            print('************')

            train_message = 'S2-down, epoch:{} lr:{}\ntrain:L1loss:{}, sam_loss:{}, psnr:{}, CC:{}, rmse:{}'. \
                format(epoch, lr,
                       np.mean(np.abs(out_lrhsi - out_lrmsi)),
                       compute_sam(out_lrmsi, out_lrhsi),
                       compute_psnr(out_lrmsi, out_lrhsi),
                       compute_cc(out_lrmsi, out_lrhsi),
                       compute_rmse(out_lrmsi, out_lrhsi)
                       )
            print(train_message)

            print('************')
            test_message_SRF = 'S2-SRF: generated lrmsifromlhsi and true lrmsifromlhsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC{}, rmse:{}'. \
                format(epoch, lr,
                       np.mean(np.abs(out_lrhsi - out_fromlrhsi_true)),
                       compute_sam(out_fromlrhsi_true, out_lrhsi),
                       compute_psnr(out_fromlrhsi_true, out_lrhsi),
                       compute_ergas(out_fromlrhsi_true, out_lrhsi, args.scale_factor),
                       compute_cc(out_fromlrhsi_true, out_lrhsi),
                       compute_rmse(out_fromlrhsi_true, out_lrhsi)
                       )
            print(test_message_SRF)

            print('************')
            test_message_PSF = 'S2-PSF: generated lrmsifrommsi and true lrmsifrommsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'. \
                format(epoch, lr,
                       np.mean(np.abs(out_frommsi_true - out_lrmsi)),
                       compute_sam(out_frommsi_true, out_lrmsi),
                       compute_psnr(out_frommsi_true, out_lrmsi),
                       compute_ergas(out_frommsi_true, out_lrmsi, args.scale_factor),
                       compute_cc(out_frommsi_true, out_lrmsi),
                       compute_rmse(out_frommsi_true, out_lrmsi)
                       )
            print(test_message_PSF)
            print('\n')

    if (epoch > args.decay_begin_epoch_stage2 - 1):
        each_decay = args.lr_stage2 / (args.epoch_stage2 - args.decay_begin_epoch_stage2 + 1)
        lr = lr - each_decay
        for param_group in optimizer_Spectral_down.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_Spatial_down.param_groups:
            param_group['lr'] = lr

print_current_precision(args, 'Stage 2')
print_current_precision(args, train_message)
print_current_precision(args, test_message_SRF)
print_current_precision(args, test_message_PSF)
print_current_precision(args, 'estimated PSF:\n{}'.format(Spatial_down_net.psf.weight.data))
print_current_precision(args, 'true PSF:\n{}'.format(psf))

temp1 = [Spectral_down_net.conv2d_list[i].weight.data.cpu().numpy()[0, :, 0, 0] for i in range(0, sp_range.shape[0])]
temp2 = [temp1[i].sum() for i in range(0, sp_range.shape[0])]
estimated_SRF = [temp1[i] / temp2[i] for i in range(0, sp_range.shape[0])]
print_current_precision(args, 'estimated SRF:\n{}'.format(estimated_SRF))
print_current_precision(args, 'true SRF:\n{}'.format(
    [sp_matrix[int(sp_range[i, 0]):int(sp_range[i, 1]) + 1, i] for i in range(0, sp_range.shape[0])]))


'''
#begin stage 3
'''
print('________________________Stage 3____________________________')

out_lrmsi_frommsi_new = out_lrmsi_frommsi.clone().detach()
print("sssssssssssssssssss", out_lrmsi_frommsi_new.shape)
out_lrmsi_fromlrhsi_new = out_lrmsi_fromlrhsi.clone().detach().requires_grad_(True)

for epoch in range(1, args.epoch_stage3 + 1):
    lr = args.lr_stage3
    optimizer_Spectral_down.zero_grad()
    optimizer_Spatial_down.zero_grad()
    optimizer_Spectral_up.zero_grad()
    optimizer_Spatial_up.zero_grad()

    #down
    out_lrmsi_fromlrhsi = Spectral_down_net(lhsi)
    out_lrmsi_frommsi = Spatial_down_net(hmsi)
    loss2 = L1Loss(out_lrmsi_fromlrhsi, out_lrmsi_frommsi)

    #up
    lrhsi = Spectral_up_net(out_lrmsi_frommsi_new)
    loss3 = L1Loss(lrhsi, lhsi)
    hrmsi = Spatial_up_net(out_lrmsi_fromlrhsi_new)
    loss4 = L1Loss(hrmsi, hmsi)

    #total_loss
    total_loss = loss2 + loss3 + loss4
    total_loss.backward()

    #output
    pre_hhsi = Spectral_up_net(hmsi)
    pre_lrhsi = Spatial_down_net(pre_hhsi)
    loss5 = L1Loss(pre_lrhsi, lhsi)
    loss5.backward()

    optimizer_Spectral_down.step()
    optimizer_Spatial_down.step()
    optimizer_Spectral_up.step()
    optimizer_Spatial_up.step()

    if epoch % 1000 == 0:

        with torch.no_grad():
            out_lrhsi = lrhsi.detach().cpu().numpy()[0].transpose(1, 2, 0)
            est_hhsi = Spectral_up_net(hmsi).detach().cpu().numpy()[0].transpose(1, 2, 0)

            test_message_specUp = 'S3-generated hhsi and true hhsi epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'. \
                format(epoch, lr,
                       np.mean(np.abs(hhsi_true - est_hhsi)),
                       compute_sam(hhsi_true, est_hhsi),
                       compute_psnr(hhsi_true, est_hhsi),
                       compute_ergas(hhsi_true, est_hhsi, args.scale_factor),
                       compute_cc(hhsi_true, est_hhsi),
                       compute_rmse(hhsi_true, est_hhsi),
                       )
            print(test_message_specUp)
            print('\n')

    if (epoch > args.decay_begin_epoch_stage3 - 1):
        each_decay = args.lr_stage3 / (args.epoch_stage3 - args.decay_begin_epoch_stage3 + 1)
        lr = lr - each_decay
        for param_group in optimizer_Spectral_up.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_Spatial_up.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_Spectral_down.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_Spatial_down.param_groups:
            param_group['lr'] = lr

est_hhsi1 = Spectral_up_net(hmsi).detach().cpu().numpy()[0].transpose(1, 2, 0)
save_wholehhsi(args, est_hhsi1)
s1s2_over_time = time.time()
s1s2_time = s1s2_over_time - s1_start_time

###store the result
print_current_precision(args, '\n')
print_current_precision(args, 'results in Stage 2')
print_current_precision(args, test_message_specUp)
print_current_precision(args, 'time of S1+S2+S3:{}'.format(s1s2_time))

from options.visual import save_net

##save trained three module
save_net(args, Spectral_up_net)
save_net(args, Spectral_down_net)
save_net(args, Spatial_down_net)
save_net(args, Spatial_up_net)







