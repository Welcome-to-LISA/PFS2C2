#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class Dataset
    Generate simulation data
~~~~~~~~~~~~~~~~~~~~~~~~
Function:
    downsamplePSF: The function of this function is to ensure that the same Gaussian downsampling method is used with matlab.


"""
import scipy.io
import torch.utils.data as data
import torch
import os
import glob
import scipy.io as io
import numpy as np
import random
# from osgeo import gdal
import sys
from dataset import return_spmatrix, get_sp_range
import xlrd
import h5py
import mat73



# def gdal_read(path):
#     '''try to open image using gdal'''
#     try:
#         img_raster = gdal.Open(path)
#         # 使用gdal读取卫星数据，gdal的一个用c++语言编写的库，用于处理地理信息相关的数据包括转换，识别数据，格式化数据以及解析
#     except RuntimeError:
#         print("Unable to open %s" % path)
#         sys.exit(1)
#
#     img_array = img_raster.ReadAsArray()
#
#     '''transpose image if the channel dimension is the first'''
#     if img_raster.RasterCount == img_array.shape[0]:
#         # 判断图像的波段数是否等于图像矩阵的行数（图像的高）
#         img_array = img_array.transpose(1, 2, 0)
#     return img_array
#
# def get_patch(img, patch_size=128):
#     ih, iw, ic = img.shape
#     tp = patch_size
#     ip = patch_size
#     ix = random.randrange(0, iw - ip - 1)
#
#     # 没有指定递增基数，返回范围内的一个随机数
#     iy = random.randrange(0, ih - ip - 1)
#     return ix, iy

class Dataset(data.Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.args = args
        #获取sp_matrix

        self.sp_matrix = return_spmatrix(self.args.data_name)
        self.sp_range = get_sp_range(self.sp_matrix)
        self.PSF = self.matlab_style_gauss2D(shape=(self.args.scale_factor, self.args.scale_factor),
                                             sigma=self.args.sigma)

        self.InvPSF = self.create_inverse_PSF(shape=(self.args.scale_factor, self.args.scale_factor),
                                              sigma=self.args.sigma)

        data_path = os.path.join("J:\\PFS2C2_real\\data", self.args.data_name)

        input_msi_path = os.path.join(data_path, "LN01_MSI.mat")
        input_hsi_path = os.path.join(data_path, "LN01_HSI.mat")
        print('load dataset')

        

        self.img_msi_list = []
        msi_data = scipy.io.loadmat(input_msi_path)["MSI"].astype("float16")
        # msi_data = mat73.loadmat(input_msi_path)["data"].astype("float16")
        # msi_data = h5py.File(input_msi_path)["data"].astype("float16")
        ##通道数在最后面
        # msi_data = np.transpose(msi_data, (1, 2, 0))
        print('shape', msi_data.shape)
        self.img_msi_list.append(self.normalize_data(msi_data))

        self.img_hsi_list = []
        hsi_data = scipy.io.loadmat(input_hsi_path)["HSI"].astype("float16")
        # hsi_data = mat73.loadmat(input_hsi_path)["data"].astype("float16")
        # hsi_data = h5py.File(input_hsi_path)["data"].astype("float16")
        # 有些数据的格式不符合这里的定义，转换一下，不是必须
        # hsi_data = np.transpose(hsi_data, (1, 2, 0))
        print('shape', hsi_data.shape)
        self.img_hsi_list.append(self.normalize_data(hsi_data))

        (_, _, self.hsi_channels) = self.img_hsi_list[0].shape
        (_, _, self.msi_channels) = self.img_msi_list[0].shape
        self.mask = self.getmask(hsi_data)
        # self.mask_x, self.mask_y = np.where(mask == True)

    def matlab_style_gauss2D(self, shape=(3, 3), sigma=2):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def create_inverse_PSF(self, sigma, shape=(3, 3)):
        """
        创建一个逆PSF核，假设原始PSF是高斯核
        这里使用高斯核的倒数来近似逆PSF核
        """
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = torch.meshgrid(torch.arange(-m, m + 1), torch.arange(-n, n + 1))
        h = torch.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def getmask(self, img):
        img_max = np.max(img, 2)
        # 返回aix=2上的最大值
        img_min = np.min(img, 2)
        #   按第三个维度对array1进行拆分，得到array1[:, :, 0]、array1[:, :, 1]、array1[:, :, 2]、array1[:, :, 3]，然后对array1[:, :, 0]、
        #   array1[:, :, 1]、array1[:, :, 2]、array1[:, :, 3]的对应元素进行逐位比较
        img_mask = (img_max == 0) & (img_min == 0)
        return img_mask

    def normalize_data(self, data):
        data = (data - np.min(data[:])) / (np.max(data[:]) - np.min(data[:]))
        return data


    def __getitem__(self, index):
        img_hsi = self.img_hsi_list[index]
        img_msi = self.img_msi_list[index]

        img_name = "ZY1E_SINGLE_IMG"

        img_tensor_lr = torch.from_numpy(img_hsi.transpose(2, 0, 1).copy()).float()
        # 浅拷贝转换矩阵维度后的张量
        img_tensor_rgb = torch.from_numpy(img_msi.transpose(2, 0, 1).copy()).float()

        return {"lhsi": img_tensor_lr,
                'hmsi': img_tensor_rgb,
                "name": img_name}

    def __len__(self):
        return len(self.img_hsi_list)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_name", type=str, default="cave")
    parser.add_argument("--scale_factor", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=32)
    args = parser.parse_args()

    train_dataset = Dataset(args, "train")
    test_lr, test_rgb, test_hr, name = train_dataset.__getitem__(0)

