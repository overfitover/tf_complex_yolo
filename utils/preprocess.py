#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# import config as cfg
import sys
from config import cfg
import numpy as np
import h5py
import scipy.misc as misc


def process_pointcloud():
    """
    input: pc (n, 3)
    output: feature_buffer 
    """
    pass


def generate_rgbmap(pointcloud):
    '''
    kitti_loader
    x [0, 40]
    y [-40, 40]
    z [-2, 1.25]

    1024 * 512
    n * m
    rslidar_loader
    [-40, 40]
    [-2, 2]
    输入点云
    返回 n * m * 3
    '''
    # 筛选范围内的点
    lidar_coord = np.array([-cfg.X_MIN, -cfg.Y_MIN, -cfg.Z_MIN], dtype=np.float32)
    voxel_size = np.array([cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE], dtype=np.float32)
    # min(1.0, log(N+1)/64)
    r_map = np.zeros([cfg.RGB_Map_M, cfg.RGB_Map_N], dtype = np.float32)
    # max z
    g_map = np.zeros([cfg.RGB_Map_M, cfg.RGB_Map_N], dtype = np.float32)
    # max i
    b_map = np.zeros([cfg.RGB_Map_M, cfg.RGB_Map_N], dtype = np.float32)

    bound_x = np.logical_and(
        pointcloud[:, 0] >= cfg.X_MIN, pointcloud[:, 0] < cfg.X_MAX)
    bound_y = np.logical_and(
        pointcloud[:, 1] >= cfg.Y_MIN, pointcloud[:, 1] < cfg.Y_MAX)
    bound_z = np.logical_and(
        pointcloud[:, 2] >= cfg.Z_MIN, pointcloud[:, 2] < cfg.Z_MAX)

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    pointcloud = pointcloud[bound_box]

    shifted_coord = pointcloud[:, :3] + lidar_coord
    voxel_index = np.floor(shifted_coord[:, 0:2] / voxel_size).astype(np.int)
    
    for idx in range(pointcloud.shape[0]):
        
        # pc_xy = pointcloud[idx][:2]
        # x = pointcloud[idx][0]
        # y = pointcloud[idx][1]
        z = pointcloud[idx][2]
        i = pointcloud[idx][3]

        index_x, index_y = voxel_index[idx]
        r_map[index_x, index_y] += 1

        if g_map[index_x, index_y] < z:
            g_map[index_x, index_y] = z
        
        if b_map[index_x, index_y] < i:
            b_map[index_x, index_y] = i

    r_map = np.log(r_map + 1) / 64.0
    r_map = np.minimum(r_map, 1.0)

    r_map = r_map[..., np.newaxis]
    g_map = g_map[..., np.newaxis]
    b_map = b_map[..., np.newaxis]

    rgb_map = np.concatenate((r_map, g_map, b_map), axis=2)
    
    return rgb_map

def val():
    binfile = '/home/yxk/data/Kitti/object/training/velodyne/000001.bin'
    pc = np.fromfile(binfile, dtype=np.float32).reshape(-1, 4)
    rgbmap = generate_rgbmap(pc)

    # f = h5py.File('rgb_map.h5', 'w')
    # f.create_dataset('rgb_map', data=rgbmap)
    # f.close()
    misc.imsave('eval_bv.png', rgbmap)

    print(rgbmap.shape)

    

if __name__ == '__main__':
    val()
