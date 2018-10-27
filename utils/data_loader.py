#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import numpy as np
import os
import sys
import glob
import multiprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import cfg
from utils import utils_tool

from utils.preprocess import generate_rgbmap
from utils.preprocess import process_pointcloud
import math
import cv2
import time
import scipy.misc as misc

class Processor:
    def __init__(self, data_tag, f_rgb, f_lidar, f_label, data_dir, aug=False, is_testset=False):
        self.data_tag = data_tag
        self.f_rgb = f_rgb
        self.f_lidar = f_lidar
        self.f_label = f_label
        self.data_dir = data_dir
        self.aug = aug
        self.is_testset = is_testset
        
    def __call__(self, load_index):
        if self.aug:
            ret = aug_data(self.data_tag[load_index], self.data_dir)
        else:
            rgb = cv2.resize(cv2.imread(self.f_rgb[load_index]), (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            raw_lidar = np.fromfile(self.f_lidar[load_index], dtype=np.float32).reshape((-1, 4))

            for i in range(raw_lidar.shape[0]):
                if np.isnan(raw_lidar[i][0]) or np.isnan(raw_lidar[i][1]) or np.isnan(raw_lidar[i][2]) or np.isnan(raw_lidar[i][3]):
                    # print('----- rm nan -----')
                    np.delete(raw_lidar, i, 0)
                if raw_lidar[i][0] == 0 and raw_lidar[i][1] == 0 and raw_lidar[i][2] == 0:
                    # print('----- rm 0 -----')
                    np.delete(raw_lidar, i, 0)
            
            if not self.is_testset:
                labels = [line for line in open(self.f_label[load_index], 'r').readlines()]
            else:
                labels = ['']
            tag = self.data_tag[load_index]
            # voxel ()
            # starttime = time.time()
            rgb_map = generate_rgbmap(raw_lidar)
            loss_feed_val = utils_tool.label_loss(labels)

            # print('loss_feed_val', loss_feed_val, type(loss_feed_val))

            ret = [tag, raw_lidar, rgb_map, labels, rgb, loss_feed_val]
        return ret

# global pool
TRAIN_POOL = multiprocessing.Pool(4)
VAL_POOL = multiprocessing.Pool(2)

def iterate_data(data_dir, shuffle=False, aug=False, is_testset=False, batch_size=1, multi_gpu_sum=1):

    f_lidar = glob.glob(os.path.join(data_dir, 'velodyne', '*.bin'))
    f_label = glob.glob(os.path.join(data_dir, 'label_2', '*.txt'))
    f_rgb = glob.glob(os.path.join(data_dir, 'image_2', '*.png'))

    f_rgb.sort()
    f_lidar.sort()
    f_label.sort()

    data_tag = [name.split('/')[-1].split('.')[-2] for name in f_lidar]

    assert len(data_tag) != 0, "dataset folder is not correct!"
    assert len(data_tag) == len(f_lidar) == len(f_label), "dataset folder is not correct!"
    
    nums = len(f_lidar)
    indices = list(range(nums))
    if shuffle:
        np.random.shuffle(indices)

    num_batches = int(math.floor(nums / float(batch_size)))
    # self, data_tag, f_lidar, f_label, data_dir, aug, is_testset
    proc = Processor(data_tag, f_rgb, f_lidar, f_label, data_dir, aug, is_testset)
    for batch_idx in range(num_batches):

        start_idx = batch_idx * batch_size
        excerpt = indices[start_idx: start_idx + batch_size]
        rets = TRAIN_POOL.map(proc, excerpt)

        tag = [ret[0] for ret in rets]
        raw_lidar = [ret[1] for ret in rets]
        rgb_map = [ret[2] for ret in rets]
        labels = [ret[3] for ret in rets]
        rgb = [ret[4] for ret in rets]
        loss_feed = [ret[5] for ret in rets]

        ret = (
               np.array(tag),
               np.array(labels),
               np.array(rgb_map),
               np.array(rgb),
               np.array(raw_lidar),
               np.array(loss_feed)
               )
        yield ret


def val1():
    data_dir = '/home/yxk/data/Kitti/object'

    a = 0
    for batch in iterate_data(data_dir + '/training'):
        tag = batch[0]
        labels = batch[1]
        rgb_map = batch[2]
        rgb = batch[3]
        raw_lidar = batch[4]
        loss_feed = batch[5]

        chunk = utils_tool.label_to_gt_box2d_chunk(labels[0], cls='All', coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None)
        print(chunk)
        print(chunk[0][0][1])


        # print(loss_feed['botright'])
        # print("tag", tag)
        # print("labels", labels)
        # print("voxel", voxel.shape)
        # print("rgb", rgb.shape)
        # print("raw_lidar", raw_lidar.shape)
        # print("loss_feed", loss_feed.dtype)

        # a = int(chunk[0][0][1])
        # b = int(chunk[0][0][2])
        # c = int(chunk[0][0][3])
        # d = int(chunk[0][0][4])
        # print(a, b, c, d)
        # misc.imsave('voxel.png', rgb_map[0])
        # img = cv2.imread('voxel.png')
        # cv2.rectangle(img, (a, b), (c, d), (255, 0, 0), 1)
        # # cv2.rectangle(img, (100, 200), (500, 600), (255, 0, 0), 1)
        # cv2.imwrite('bb.png', img)


        a += 1
        if a > 0:
            break


if __name__ == '__main__':
    val1()
