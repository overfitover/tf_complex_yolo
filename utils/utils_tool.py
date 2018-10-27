# -*- cooing:UTF-8 -*-
import numpy as np
from config import cfg
from copy import deepcopy

def cal_anchors():
    '''

    '''
    pass

def cal_targets():
    '''
    
    '''
    pass


def labelbatch(chunk):
    """
    description: 处理 label 产生loss_layer层的输入
    (N, N' labels)
    Takes a chunk of parsed annotations

    returns value for placeholders of net's 
    loss layer correspond to this chunk
    """
    W, H = cfg.CELL_SIZE_X, cfg.CELL_SIZE_Y     # 16  32
    B, C = cfg.NUM_ANCHORS, cfg.NUM_CLASSES     # 5   8

    w = cfg.VOXEL_X   # 40
    h = cfg.VOXEL_Y   # 80

    # jpg = chunk[0];
    allobj = deepcopy(chunk)

    # Calculate regression target
    cellx = 1. * w / W
    celly = 1. * h / H

    allobj = np.reshape(allobj, (-1, 7))
    allobj_new = []    # 扩展了列表到八位
    # cls xmin ymin xmax ymax im re
    for obj in allobj:
        centerx = .5*(obj[1]+obj[3])   # xmin, xmax
        centery = .5*(obj[2]+obj[4])   # ymin, ymax
        
        cx = centerx / cellx
        cy = centery / celly
        if cx >= W or cy >= H: continue
        obj[3] = float(obj[3]-obj[1]) / w       # w 比例值
        obj[4] = float(obj[4]-obj[2]) / h       # h 比例值
        obj[3] = np.sqrt(obj[3])                # 开方
        obj[4] = np.sqrt(obj[4])

        obj[1] = cx - np.floor(cx)  # centerx  占每一cell的比例值
        obj[2] = cy - np.floor(cy)  # centery
        obj = list(obj)
        obj += [int(np.floor(cy) * W + np.floor(cx))]   # 把label坐标转换成一维数组保存下来
        allobj_new.append(obj)

    # print(np.shape(allobj_new))
    # show(im, allobj, S, w, h, cellx, celly) # unit test
    # Calculate placeholders' values
    probs = np.zeros([H*W, B, C])
    confs = np.zeros([H*W, B])
    coord = np.zeros([H*W, B, 4])
    euler = np.zeros([H*W, B, 2])
    proid = np.zeros([H*W, B, C])
    prear = np.zeros([H*W, 4])

    for obj in allobj_new:
        probs[obj[7], :, :] = [[0.]*C] * B              # (B, C)  obj[7] 表示object位置
        probs[obj[7], :, int(obj[0])] = 1.              # label 位置有某个物体的值是1

        proid[obj[7], :, :] = [[1.]*C] * B

        coord[obj[7], :, :] = [obj[1:5]] * B            # 考虑不同的anchor

        prear[obj[7], 0] = (obj[1]+obj[7] % 16)*cellx - obj[3]**2 * .5 * W   # xleft
        prear[obj[7], 1] = (obj[2]+obj[7] // 16)*celly - obj[4]**2 * .5 * H  # yup
        prear[obj[7], 2] = (obj[1]+obj[7] % 16)*cellx + obj[3]**2 * .5 * W   # xright
        prear[obj[7], 3] = (obj[2]+obj[7] // 16)*celly + obj[4]**2 * .5 * H  # ybot

        confs[obj[7], :] = [1.] * B                     # 有物体的位置是1

        # tim = sin   tre = cos
        euler[obj[7], :, 0] = obj[5]  
        euler[obj[7], :, 1] = obj[6]


    # Finalise the placeholders' values
    upleft = np.expand_dims(prear[:, 0:2], 1)
    botright = np.expand_dims(prear[:, 2:4], 1)
    wh = botright - upleft
    area = wh[:, :, 0] * wh[:, :, 1]
    upleft = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at loss layer 
    loss_feed_val = {
        'probs': probs, 'confs': confs, 
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft, 
        'botright': botright, 'euler': euler
    }
    # print('loss_feed_val--', loss_feed_val, type(loss_feed_val))

    return loss_feed_val

def label_loss(labels):
    '''
    description: 计算label的loss输入
    :return:
    'probs': probs, 'confs': confs,
    'coord': coord, 'proid': proid,
    'areas': areas, 'upleft': upleft,
    'botright': botright, 'euler': euler
    '''

    chunk = label_to_gt_box2d_chunk(labels, cls='All', coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None)
    # print('chunk: ', chunk, type(chunk), np.shape(chunk))
    loss_feed_val = labelbatch(chunk)
    # print('loss_feed_val: ', loss_feed_val, type(loss_feed_val), np.shape(loss_feed_val))
    return loss_feed_val

def label_to_gt_box3d(labels, cls='All', coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   label: (N, N')
    #   cls: 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate: 'camera' or 'lidar'
    # Output:
    #   (N, N', 7)
    boxes3d = []
    if cls == 'Car':
        acc_cls = ['Car', 'Van']
    elif cls == 'Pedestrian':
        acc_cls = ['Pedestrian']
    elif cls == 'Cyclist':
        acc_cls = ['Cyclist']
    else: # all
        acc_cls = cfg.LABELNAMES

    for label in labels:
        boxes3d_a_label = []
        for line in label:
            ret = line.split()
            if ret[0] in acc_cls or acc_cls == []:
                h, w, l, x, y, z, r = [float(i) for i in ret[-7:]]
                box3d = np.array([x, y, z, h, w, l, r])
                boxes3d_a_label.append(box3d)
        if coordinate == 'lidar':
            # 去除坐标转换
            boxes3d_a_label = camera_to_lidar_box(np.array(boxes3d_a_label), T_VELO_2_CAM, R_RECT_0)

        boxes3d.append(np.array(boxes3d_a_label).reshape(-1, 7))
    return boxes3d


def camera_to_lidar(x, y, z, T_VELO_2_CAM=None, R_RECT_0=None):
    '''
    :param x:
    :param y:
    :param z:
    :param T_VELO_2_CAM:
    :param R_RECT_0:
    :return: lidar x y z
    '''
    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(cfg.MATRIX_T_VELO_2_CAM)

    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(cfg.MATRIX_R_RECT_0)

    p = np.array([x, y, z, 1])
    p = np.matmul(np.linalg.inv(R_RECT_0), p)
    p = np.matmul(np.linalg.inv(T_VELO_2_CAM), p)
    p = p[0:3]
    return tuple(p)

#camera_to_lidar(float(-0.69), 1.69, 25.01, T_VELO_2_CAM=None, R_RECT_0=None)

def camera_to_lidar_box(boxes, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 8) -> (N, 8) category,x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        category, x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(
            x, y, z, T_VELO_2_CAM, R_RECT_0), h, w, l, -ry - np.pi / 2
        rz = angle_in_limit(rz)
        ret.append([category, x, y, z, h, w, l, rz])
    return np.array(ret).reshape(-1, 8)

def label_to_gt_box2d_chunk(labels, cls='All', coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   label: (N, N')
    #   cls: 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate: 'camera' or 'lidar'
    # Output:  category, xmin, ymin, xmax, ymax, im, re
    #   (N, N', 7)
    # cls， xmin

    boxes2d = []

    labelnames = cfg.LABELNAMES
    if cls == 'Car':
        acc_cls = ['Car', 'Van']
    elif cls == 'Pedestrian':
        acc_cls = ['Pedestrian']
    elif cls == 'Cyclist':
        acc_cls = ['Cyclist']
    else:  # all
        acc_cls = cfg.LABELNAMES

    boxes3d_a_label = []
    label_category = []
    for line in labels:
        ret = line.split()
        category = ret[0]

        if category in acc_cls or acc_cls == []:
            h, w, l, x, y, z, r = [float(i) for i in ret[-7:]]

            catetype = labelnames.index(category)
            box3d = np.array([catetype, x, y, z, h, w, l, r])
            boxes3d_a_label.append(box3d)
    if coordinate == 'lidar':
        # 去除坐标转换
        boxes3d_a_label = camera_to_lidar_box(np.array(boxes3d_a_label), T_VELO_2_CAM, R_RECT_0)

    # boxes3d_a_label 2d
    boxes2d_a_label = box3d_to_box2d(boxes3d_a_label)
    boxes2d.append(np.array(boxes2d_a_label).reshape(-1, 7))

    '''
    for label in labels:
        boxes3d_a_label = []
        label_category = []
        print('label =', label)
        for line in label:
            ret = line.split()
            category = ret[0]
            print('ret =', ret)
            if category in acc_cls or acc_cls == []:
                h, w, l, x, y, z, r = [float(i) for i in ret[-7:]]

                catetype = labelnames.index(category)
                box3d = np.array([catetype, x, y, z, h, w, l, r])
                boxes3d_a_label.append(box3d)
        if coordinate == 'lidar':
            # 去除坐标转换
            boxes3d_a_label = camera_to_lidar_box(np.array(boxes3d_a_label), T_VELO_2_CAM, R_RECT_0)

        # boxes3d_a_label 2d
        boxes2d_a_label = box3d_to_box2d(boxes3d_a_label)
        boxes2d.append(np.array(boxes2d_a_label).reshape(-1, 7))
    '''
    return boxes2d

def box3d_to_box2d(boxes):
    '''
    :param boxes: (N, 8) category,x,y,z,h,w,l,r
    :return: (N, 7)   category, xmin, ymin, xmax, ymax, im, re
    '''
    ret = []
    for box in boxes:
        category, x, y, z, h, w, l, ry = box
        xmin = x - w/2.
        xmax = x + w/2.
        y = cfg.Y_MAX - y
        ymin = y - h/2.
        ymax = y + h/2.
        im = np.sin(ry)
        re = np.cos(ry)
        ret.append([category, xmin, ymin, xmax, ymax, im, re])
    return ret

def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle

# print(camera_to_lidar(1, 2, 3, T_VELO_2_CAM=None, R_RECT_0=None))

def lidar_to_bird_view_img(lidar, factor=1):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    birdview = np.zeros(
        (cfg.INPUT_HEIGHT * factor, cfg.INPUT_WIDTH * factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if cfg.X_MIN < x < cfg.X_MAX and cfg.Y_MIN < y < cfg.Y_MAX:
            x, y = int((x - cfg.X_MIN) / cfg.VOXEL_X_SIZE *
                       factor), int((y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # divisor = 100
    # TODO: adjust this factor
    birdview = np.clip((birdview / divisor * 255) *
                       5 * factor, a_min=0, a_max=255)
    birdview = np.tile(birdview, 3).astype(np.uint8)

    return birdview