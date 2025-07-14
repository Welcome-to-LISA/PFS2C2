#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xlrd
import numpy as np
import os
import torch
import glob
import scipy.io as io


sp_root_path = "../../data/"
estimated_R_root_path = '../../data/EstimatedR'
sp_path = "../../data/paviau/"

def get_spectral_response(data_name):
    xls_path = os.path.join(sp_path, data_name + '.xls')
    print(xls_path)
    if not os.path.exists(xls_path):
        raise Exception("spectral response path does not exist")
    data = xlrd.open_workbook(xls_path)
    table = data.sheets()[0]
    num_cols = table.ncols
    cols_list = [np.array(table.col_values(i)).reshape(-1,1) for i in range(1,num_cols)]

    sp_data = np.concatenate(cols_list, axis=1)
    sp_data = sp_data / (sp_data.sum(axis=0))

    return sp_data

def get_msi_wavelength_range(data_name):
    wl_xls_path = os.path.join("J://PFS2C2_real//data//spectral_response", data_name, 'msi_wl.xls')
    print('get_msi_wl_xls')
    if not os.path.exists(wl_xls_path):
        raise Exception("wavelength path does not exist")
    wl_data = xlrd.open_workbook(wl_xls_path)
    wl_table = wl_data.sheets()[0]
    num_cols = wl_table.ncols
    cols_list = [np.array(wl_table.col_values(i)).reshape(-1, 1) for i in range(0, num_cols)]
    wl = np.concatenate(cols_list, axis=1)
    return wl.astype('float64')

def get_hsi_wavelength(data_name):
    wl_xls_path = os.path.join(r"J://PFS2C2_real//data//spectral_response", data_name, 'hsi_wl.xls')
    if not os.path.exists(wl_xls_path):
        print(wl_xls_path)
        raise Exception("wavelength path does not exist")
    wl_data = xlrd.open_workbook(wl_xls_path)
    wl_table = wl_data.sheets()[0]
    num_cols = wl_table.ncols
    cols_list = [np.array(wl_table.col_values(i)).reshape(-1, 1) for i in range(0, num_cols)]
    wl = np.concatenate(cols_list, axis=1)
    return wl.astype('float64')

def generate_fake_srf(msi_wl_range):
    start_wl = 400
    end_wl = 2500
    msi_wl_range = np.transpose(msi_wl_range)
    num_bands = msi_wl_range.shape[1]
    print(num_bands)
    zero_data = np.zeros((end_wl - start_wl, num_bands + 1))
    zero_data[:, 0] = np.linspace(start_wl, end_wl, num=end_wl - start_wl + 1)[:-1]
    for i in range(num_bands):
        sigma = 1
        mu = np.mean(msi_wl_range[:, i])
        s_wl = int(msi_wl_range[0, i])
        e_wl = int(msi_wl_range[1, i])
        x = np.linspace(s_wl, e_wl, num=e_wl - s_wl + 1)
        y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        # y = np.reshape(y, [-1, 1])
        zero_data[s_wl:e_wl + 1, i + 1] = y
    return zero_data

def get_spectral_R_estimated(data_name, sigma):
    path_list = glob.glob(os.path.join(estimated_R_root_path, '*.mat'))
    path_list_lower = [filename.lower() for filename in path_list]
    for index, path in enumerate(path_list_lower):
        if sigma > 1:
            sigma = int(sigma)
        if path.find(data_name) >= 0 and path.find(str(sigma)) >= 0:
            estimated_R = io.loadmat(path_list[index])['R'].transpose(1, 0)
            break
    return estimated_R

# def create_dataset(arg, sp_matrix):
#     dataset_instance = Dataset(arg, sp_matrix)
#     return dataset_instance

def get_sp_range(sp_matrix):

    HSI_bands, MSI_bands = sp_matrix.shape
    print( "输出波段范伟了", HSI_bands, MSI_bands)
    assert (HSI_bands > MSI_bands)
    sp_range = np.zeros([MSI_bands, 2])
    for i in range(0, MSI_bands):
        index_dim_0, index_dim_1 = np.where(sp_matrix[:, i].reshape(-1, 1) > 0)
        # 判断大于0的数值下标记录下来，变成一列，不明白dim_1的作用按理说他全是0
        sp_range[i, 0] = index_dim_0[0]
        sp_range[i, 1] = index_dim_0[-1]
    print(sp_range)

    return sp_range

def get_srf(sat_name):
    xls_path = os.path.join("../data/", sat_name + '.xls')
    if not os.path.exists(xls_path):
        raise Exception("spectral response path does not exist")
    data = xlrd.open_workbook(xls_path)
    table = data.sheets()[0]
    num_cols = table.ncols
    # 获取列数
    cols_list = [np.array(table.col_values(i)).reshape(-1, 1) for i in range(0, num_cols)]
    srf = np.concatenate(cols_list, axis=1)
    # 将两个二维数组拼接在一起
    return srf

# 用于计算目标波长下的模拟响应函数。
def cal_simul_srf(tar_wl, sat_srf):
    sat_wl = sat_srf[1:, 0].reshape(1, -1).astype("float64")
    # 取矩阵中从第一行开始的所有0列元素并转为一行
    sat_srf_value = sat_srf[1:, 1:].T.astype("float64")
    min_value = tar_wl - sat_wl
    min_index = np.argmin(np.absolute(min_value), axis=1)
    # 返回维度为1，绝对值最小的索引
    tar_srf = sat_srf_value[:, min_index].T
    # 将min_index所在的列转置
    tar_srf_sum = np.sum(tar_srf, axis=0)
    if np.any(tar_srf_sum == 0):
        # 如果矩阵中有任意一个元素为0，则执行
        mask = (tar_srf_sum != 0)
        tar_srf = tar_srf[:, mask]
    return tar_srf / (tar_srf.sum(axis=0))


def return_spmatrix(data_name):

    tar_wl = get_hsi_wavelength(data_name)
    print("print(self.msi_wl_range)", tar_wl.shape)
    msi_wl_range = get_msi_wavelength_range(data_name)
    print("msi_wl_range", msi_wl_range.shape)
    sat_srf = generate_fake_srf(msi_wl_range)
    print("ssss", sat_srf.shape)
    srf = cal_simul_srf(tar_wl, sat_srf)

    return srf







