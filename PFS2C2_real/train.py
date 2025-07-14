# -*- coding: utf-8 -*-
"""

"""
import torch
import torch.nn as nn
from evalution import compute_sam, compute_psnr, compute_ergas, compute_cc, compute_rmse
import numpy as np
import random
from option.visual import print_current_precision, print_options
import matplotlib.pyplot as plt

from model.spectral_upsample import Spectral_upsample
from model.spectral_downsample import Spectral_downsample
from model.spatial_downsample import Spatial_downsample
from model.spatial_upsample import Spatial_upsample

from option.config import args
from dataset.dataloader import Dataset
import time
import scipy.io as io

###save estimated HHSI
from option.visual import save_hhsi, save_wholehhsi


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2)  # seed is set to 2

train_dataset = Dataset(args)
print("seddddfrergsrtgsdfgs")
hsi_channels = int(train_dataset.hsi_channels)
msi_channels = int(train_dataset.msi_channels)
sp_range = train_dataset.sp_range
sp_matrix = train_dataset.sp_matrix
psf = train_dataset.PSF
invpsf = train_dataset.InvPSF

# store the training configuration in opt.txt
print_options(args)

lhsi = train_dataset[0]["lhsi"].unsqueeze(0).to(args.device)
hmsi = train_dataset[0]['hmsi'].unsqueeze(0).to(args.device)


# reference 3-order H,W,C
out_lrhsi_true = lhsi.detach().cpu().numpy()[0].transpose(1, 2, 0)
print("sdfsdfer",out_lrhsi_true.shape)
out_msi_true = hmsi.detach().cpu().numpy()[0].transpose(1, 2, 0)
print("sdfsd",msi_channels)
print("dfds",hsi_channels)

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
print('________________________stage1____________________________')
##S1 start
s1_start_time = time.time()
for epoch in range(1, args.epoch_stage1 + 1):
    lr = args.lr_stage1
    if(epoch%1000==0):
          print(epoch)

    optimizer_Spatial_up.zero_grad()
    optimizer_Spectral_up.zero_grad()
    out_hrhsi_fromlrhsi = Spatial_up_net(lhsi)  # spectrally degraded from lrhsi
    out_hrmsi_frommsi = Spectral_up_net(hmsi)  # spatially degraded from hrmsi
    loss0 = L1Loss(out_hrhsi_fromlrhsi, out_hrmsi_frommsi)
    loss0.backward()
    optimizer_Spatial_up.step()
    optimizer_Spectral_up.step()

    if (epoch > args.decay_begin_epoch_stage1 - 1):
        each_decay = args.lr_stage1 / (args.epoch_stage1 - args.decay_begin_epoch_stage1 + 1)
        lr = lr - each_decay
        for param_group in optimizer_Spatial_up.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_Spectral_up.param_groups:
            param_group['lr'] = lr

print_current_precision(args, 'Stage 1')

'''
#begin stage 2
'''
print('________________________Stage2____________________________')

for epoch in range(1, args.epoch_stage2 + 1):
    if(epoch%1000==0):
          print(epoch)
    lr = args.lr_stage2
    optimizer_Spatial_down.zero_grad()
    optimizer_Spectral_down.zero_grad()
    out_lrmsi_fromlrhsi = Spectral_down_net(lhsi)  # spectrally degraded from lrhsi
    out_lrmsi_frommsi = Spatial_down_net(hmsi)  # spatially degraded from hrmsi
    loss1 = L1Loss(out_lrmsi_fromlrhsi, out_lrmsi_frommsi)
    loss1.backward()
    optimizer_Spatial_down.step()
    optimizer_Spectral_down.step()

    if (epoch > args.decay_begin_epoch_stage1 - 1):
        each_decay = args.lr_stage1 / (args.epoch_stage1 - args.decay_begin_epoch_stage1 + 1)
        lr = lr - each_decay
        for param_group in optimizer_Spectral_down.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_Spatial_down.param_groups:
            param_group['lr'] = lr


print_current_precision(args, 'Stage 2')

'''
#begin stage 3
'''
print('________________________Stage 3____________________________')

out_lrmsi_frommsi_new = out_lrmsi_frommsi.clone().detach()
print("sssssssssssssssssss", out_lrmsi_frommsi_new.shape)
out_lrmsi_fromlrhsi_new = out_lrmsi_fromlrhsi.clone().detach().requires_grad_(True)
for epoch in range(1, args.epoch_stage3 + 1):
    if(epoch%1000==0):
          print(epoch)
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

    if(epoch%1000==0):
        est_hhsi1 = Spectral_up_net(hmsi).detach().cpu().numpy()[0].transpose(1, 2, 0)
        save_wholehhsi(args, est_hhsi1, epoch)

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
save_wholehhsi(args, est_hhsi1, epoch)
s1s2_over_time = time.time()
s1s2_time = s1s2_over_time - s1_start_time








