
from option.config import args
from model.spectral_upsample import *
import torch
import scipy.io as io
import numpy as np
import mat73
import os

spe_up = Spectral_upsample(args, 8, 144)

# sta_spe_up = torch.load('./checkpoints/ZY1E/spectral_upsample.pth', map_location=torch.device('cpu'))

sta_spe_up = torch.load('./checkpoints/ZY1E/Spectral_up_net_15000.pth', map_location=torch.device('cpu'))
spe_up.load_state_dict({k.replace('module.', ''): v for k, v in sta_spe_up.items()})



# new_spe_up = {}
#
# for k, v in sta_spe_up.items():
#     if k.startswith("module."):
#         new_spe_up[k] = v
#     else:
#         new_key = "module." + k  # 添加前缀 "module."
#         new_spe_up[new_key] = v
# spe_up.load_state_dict(new_spe_up)

if __name__ == '__main__':

    data_pth = "./data/ZY1E/msi.mat"

    # msi_data = mat73.loadmat(data_pth)["data"].astype("float32")
    msi_data = io.loadmat(data_pth)["data"].astype("float32")

    msi_tensor_s = torch.tensor(msi_data)
    print(msi_data.shape)

    msi_tensor =msi_tensor_s.permute(2,1,0).to('cuda')

    print("sdfsd",msi_data.shape)

    hhsi_data = spe_up(msi_tensor)
    print("sdffsfd",hhsi_data.shape)


    # 将 PyTorch 张量转换为 NumPy 数组
    hhsi_data_numpy = hhsi_data.detach().cpu().numpy()

    # io.savemat('G:\\Liaohe_ahhsi.mat', {'data':  hhsi_data_numpy})

    io.savemat('G:\\Liaohe_msi.mat', {'data': hhsi_data_numpy})

    print("over")