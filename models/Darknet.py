import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from config import cfg
from models.functions import leaky_relu
from utils.utils_tool import *


# GNU lisense
# reference
# https://github.com/ruiminshen/yolo-tf/blob/master/model/yolo2/inference.py
# https://github.com/thtrieu/darkflow/blob/master/darkflow/net/yolov2/train.py
def expit_tensor(x):
    return 1. / (1. + tf.exp(-x))

class Darknet(object):
    def __init__(self, classes, num_anchors, is_training, batch_size, center=True):
        self.INPUT_M = cfg.RGB_Map_M  # 512
        self.INPUT_N = cfg.RGB_Map_N  # 1024

        self.cell_size_x = cfg.CELL_SIZE_X  # 16
        self.cell_size_y = cfg.CELL_SIZE_Y  # 32
        self.center = center
        self.FEATURE_NUM = cfg.FEATURE_NUM  # 3
        self.alpha = cfg.ALPHA  # 0.1

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE
        self.coord_euler_scale = cfg.COORD_EULER_SCALE

        self.batch_size = batch_size
        self.boxes_per_cell = cfg.NUM_ANCHORS      # 5
        # self.classes = cfg.NUM_CLASSES           # 8
        self.classes = classes
        self.num_anchors = num_anchors             # 5
        self.anchors = cfg.ANCHORS

        self.loss = None
        self.input = tf.placeholder(tf.float32, shape=[self.batch_size, self.INPUT_M, self.INPUT_N, self.FEATURE_NUM])       # batch_size, 512, 1024, 3

        W, H = self.cell_size_x, self.cell_size_y    # 16 32
        B, C = self.boxes_per_cell, self.classes     # 5 8
        HW = H * W  # grid cell number
        size1 = [self.batch_size, HW, B, C]
        size2 = [self.batch_size, HW, B]

        self._probs = tf.placeholder(tf.float32, size1)        # data_loder 中的返回结果 有物体置一   # batch_size, 16*32, 5, 8
        self._confs = tf.placeholder(tf.float32, size2)        # 有物体 anchor 置1                  # batch_size, 16*32, 5
        self._coord = tf.placeholder(tf.float32, size2 + [4])                                      # batch_size, 16*32, 5, 4
        self._areas = tf.placeholder(tf.float32, size2)                                            # batch_size, 16*32, 5
        self._upleft = tf.placeholder(tf.float32, size2 + [2])                                     # batch_size, 16*32, 5, 2
        self._botright = tf.placeholder(tf.float32, size2 + [2])                                   # batch_size, 16*32, 5, 2
        self._euler = tf.placeholder(tf.float32, size2 + [2])                                      # batch_size, 16*32, 5, 2

        # 前向网络
        scope = 'darknet'
        net = tf.identity(self.input, name='%s/input' % scope)
        with slim.arg_scope([slim.layers.conv2d],
                            kernel_size=[3, 3],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            normalizer_fn=batch_norm,
                            activation_fn=tf.nn.leaky_relu), \
             slim.arg_scope([slim.layers.max_pool2d],
                            kernel_size=[2, 2],
                            padding='SAME'):
            index = 0
            channels = 24
            for _ in range(2):
                net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
                net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
                index += 1
                channels *= 2
            # c = 96
            channels -= 32
            # c = 64
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
            # c = 128
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels / 2, scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
            # c = 256
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            index += 1
            passthrough = tf.identity(net, name=scope + '/passthrough')
            net = slim.layers.conv2d(net, channels, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
            index += 1
            channels *= 2
            # c = 512
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
            index += 1
            channels *= 2
            # c = 1024
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            index += 1
            with tf.name_scope(scope):
                _net = reorg(passthrough)
            net = tf.concat([_net, net], 3, name='%s/concat%d' % (scope, index))
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        net = slim.layers.conv2d(net, num_anchors * (7 + classes), kernel_size=[1, 1], activation_fn=None,
                                 scope='%s/conv' % scope)
        self.out_net = tf.identity(net, name='%s/output' % scope)

        # loss计算
        self.loss = loss(net, self._probs, self._confs, self._coord, self._euler, self._areas, self._upleft, self._botright, scope='loss')
        # print(self.loss)

def reorg(net, stride=2, name='reorg'):
    '''
    @description: 重组张量（b,h,w,ch）--(b,h/2,w/2,ch*2*2)
    :param net: （b,h,w,ch）
    :param stride: 2
    :param name:
    :return: (b,h/2,w/2,ch*2*2)
    '''
    batch_size, height, width, channels = net.get_shape().as_list()
    _height, _width, _channel = height // stride, width // stride, channels * stride * stride

    with tf.name_scope(name) as name:
        net = tf.reshape(net, [batch_size, _height, stride, _width, stride, channels])
        net = tf.transpose(net, [0, 1, 3, 2, 4, 5])  # batch_size, _height, _width, stride, stride, channels
        net = tf.reshape(net, [batch_size, _height, _width, -1], name=name)
    return net

def batch_norm(net):
    net = slim.batch_norm(net, center=True, scale=True)
    return net

def darknet_yolov2(net, classes, num_anchors, training=False, center=True):
    scope = 'darknet'
    net = tf.identity(net, name='%s/input' % scope)
    with slim.arg_scope([slim.layers.conv2d],
                        kernel_size=[3, 3],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        normalizer_fn=batch_norm,
                        activation_fn=tf.nn.leaky_relu), \
         slim.arg_scope([slim.layers.max_pool2d],
                        kernel_size=[2, 2],
                        padding='SAME'):
        index = 0
        channels = 24
        for _ in range(2):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        # c = 96
        channels -= 32
        # c = 64
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        channels *= 2
        # c = 128
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        channels *= 2
        # c = 256
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        passthrough = tf.identity(net, name=scope + '/passthrough')
        net = slim.layers.conv2d(net, channels, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        channels *= 2
        # c = 512
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        channels *= 2
        # c = 1024
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        with tf.name_scope(scope):
            _net = reorg(passthrough)
        net = tf.concat([_net, net], 3, name='%s/concat%d' % (scope, index))
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
    net = slim.layers.conv2d(net, num_anchors * (7 + classes), kernel_size=[1, 1], activation_fn=None,
                             scope='%s/conv' % scope)
    net = tf.identity(net, name='%s/output' % scope)

    return scope, net

def build_target():
    pass

def loss_layer(net_out, scope='loss_layer'):
    '''
    net_out
        (None, CELL_SIZE_X, CELL_SIZE_Y, (6 + 5 + 1)*5)
        (None, CELL_SIZE_X, CELL_SIZE_Y, (6 + 8 + 1)*5)
    labels:
        # (None, N', 6(xyzlwhr))
        (None, N', 6(xyzlwhr))
    '''
    fetch = []
    with tf.variable_scope(scope):
        # 1 + 8 + 4 + 2 = 15

        W, H = cfg.CELL_SIZE_X, cfg.CELL_SIZE_Y  # 16 32
        B, C = cfg.NUM_ANCHORS, cfg.NUM_CLASSES   # 5 8

        HW = H * W  # grid cell number

        size1 = [None, HW, B, C]
        size2 = [None, HW, B]

        _probs = tf.placeholder(tf.float32, size1)  # label 这些label怎么算 utils里面函数计算
        _confs = tf.placeholder(tf.float32, size2)
        _coord = tf.placeholder(tf.float32, size2 + [4])
        _euler = tf.placeholder(tf.float32, size2 + [2])

        # weights term for L2 loss
        # _proid = tf.placeholder(tf.float32, size1)
        # material calculating IOU label
        _areas = tf.placeholder(tf.float32, size2)
        _upleft = tf.placeholder(tf.float32, size2 + [2])
        _botright = tf.placeholder(tf.float32, size2 + [2])

        net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (6 + 1 + C)])
        coords = net_out_reshape[..., :4]  # x y w h
        eulers = net_out_reshape[..., 4:6]  # im re
        anchors = cfg.ANCHORS

        coords = tf.reshape(coords, [-1, H * W, B, 4])
        adjusted_coords_xy = expit_tensor(coords[..., 0:2])  # sigma(x)
        adjusted_coords_wh = tf.sqrt(
            tf.exp(coords[..., 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
        coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

        adjusted_euler = tf.reshape(eulers, [-1, H * W, B, 2])

        adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 6])  # confidence  sigma(c)
        adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B, 1])

        adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 7:])  # category
        adjusted_prob = tf.reshape(adjusted_prob, [-1, H * W, B, C])

        adjusted_net_out = tf.concat(
            [adjusted_coords_xy, adjusted_coords_wh, adjusted_euler, adjusted_c, adjusted_prob], 3)

        wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
        area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]

        centers = coords[:, :, :, 0:2]  # x, y
        floor = centers - (wh * .5)  # 左上
        ceil = centers + (wh * .5)  # 右下

        # 计算 intersection area
        intersect_upleft = tf.maximum(floor, _upleft)
        intersect_botright = tf.minimum(ceil, _botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

        # calculate the best IOU, set 0.0 confidence for worse boxes
        iou = tf.truediv(intersect, _areas + area_pred - intersect)
        best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
        best_box = tf.to_float(best_box)
        confs = tf.multiply(best_box, _confs)  # 选择最大的iou

        # class_loss
        # object_loss
        # noobject_loss
        # coord_loss
        # eular_loss

        sprob = cfg.CLASS_SCALE
        sconf = cfg.OBJECT_SCALE
        snoob = cfg.NOOBJECT_SCALE
        scoor = cfg.COORD_SCALE
        seuler = cfg.COORD_EULER_SCALE

        conid = snoob * (1. - confs) + sconf * confs  # object_loss  noobject_loss
        weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
        cooid = scoor * weight_coo  # coord_loss

        weight_eular = tf.concat(2 * [tf.expand_dims(confs, -1)], 3)
        eular = seuler * weight_eular

        weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
        proid = sprob * weight_pro  # class_loss

        fetch += [_probs, confs, conid, cooid, eular, proid]  # 应该还有 eular

        true = tf.concat([_coord, _euler, tf.expand_dims(confs, 3), _probs], 3)
        wght = tf.concat([cooid, eular, tf.expand_dims(conid, 3), proid], 3)

        # print('Building {} loss'.format('darknet'))
        loss = tf.pow(adjusted_net_out - true, 2)
        loss = tf.multiply(loss, wght)
        loss = tf.reshape(loss, [-1, H * W * B * (6 + 1 + C)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss)
        tf.summary.scalar('{} loss'.format(''), loss)
        return loss

def calc_target(label, pred):
    """
    input:
        labels:()
        pred:
    Output:
        object_mask [n*m*5*1]
    """
    pass

def area( boxlist, scope=None):
    """Computes area of boxes.
    Args:
    boxlist: BoxList holding N boxes
    scope: name scope.
    Returns:
    a tensor with shape [N] representing box areas.
    """
    with tf.name_scope(scope, 'Area'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.get(), num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])  # 删除shape1上的一 可能有问题

def intersection(boxlist1, boxlist2, scope=None):
    """Compute pairwise intersection areas between boxes.

    Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

    Returns:
    a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'Intersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            value=boxlist1.get(), num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            value=boxlist2.get(), num_or_size_splits=4, axis=1)

        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))  # transpose 是不是没用
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(
            0.0, all_pairs_min_ymax - all_pairs_max_ymin)

        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(
            0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths

def iou(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-union between box collections.

    Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

    Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'IOU'):
        intersections = intersection(boxlist1, boxlist2)
        areas1 = area(boxlist1)
        areas2 = area(boxlist2)
        unions = (
                tf.expand_dims(
                    areas1,
                    1) +
                tf.expand_dims(
                    areas2,
                    0) -
                intersections)  # 广播
        return tf.where(
            tf.equal(intersections, 0.0),
            tf.zeros_like(intersections), tf.truediv(intersections, unions))

def calc_iou(boxes1, boxes2, scope='iou'):
    """calculate ious
    Args:
      boxes1: 4-D tensor [CELL_SIZE_X, CELL_SIZE_Y, BOXES_PER_CELL, 4]  ===> (x_center, y_center, w, h)
      boxes2: 1-D tensor [CELL_SIZE_X, CELL_SIZE_Y, BOXES_PER_CELL, 4]  ===> (x_center, y_center, w, h)
    Return:
      iou: 3-D tensor [
         CELL_SIZE_X , CELL_SIZE_Y, BOXES_PER_CELL]
    """
    with tf.variable_scope(scope):
        # 转换成 角点坐标
        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                           boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])  # xmin ymin xmax ymax
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])  # [CELL_SIZE_X, CELL_SIZE_Y, BOXES_PER_CELL, 4, 4]

        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                           boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])  # 相交的lu
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)  # 相交部分的值
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                  (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                  (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def loss1(net_out, probs, confs, coord, euler, areas, upleft, botright, scope='loss'):
    '''
    net_out
        (None, CELL_SIZE_X, CELL_SIZE_Y, (6 + 5 + 1)*5)
        (None, CELL_SIZE_X, CELL_SIZE_Y, (6 + 8 + 1)*5)
    labels:
        # (None, N', 6(xyzlwhr))
        (None, N', 6(xyzlwhr))
    '''
    fetch = []
    with tf.variable_scope(scope):
        # 1 + 8 + 4 + 2 = 15
        W, H = cfg.CELL_SIZE_X, cfg.CELL_SIZE_Y  # 16 32
        B, C = cfg.NUM_ANCHORS, cfg.NUM_CLASSES   # 5 8

        _probs = probs
        _confs = confs
        _coord = coord
        _euler = euler
        _areas = areas
        _upleft = upleft
        _botright = botright

        net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (6 + 1 + C)])
        coords = net_out_reshape[..., :4]   # x y w h
        eulers = net_out_reshape[..., 4:6]  # im re
        anchors = cfg.ANCHORS

        coords = tf.reshape(coords, [-1, H * W, B, 4])
        adjusted_coords_xy = expit_tensor(coords[..., 0:2])  # sigma(x) y

        adjusted_coords_wh = tf.sqrt(
            tf.exp(coords[..., 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))  # w h
        adjusted_coords_wh = expit_tensor(adjusted_coords_wh)

        coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)   # x y w h

        adjusted_euler = tf.reshape(eulers, [-1, H * W, B, 2])          # im re

        adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 6])  # confidence  sigma(c)
        adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B, 1])

        adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 7:])  # category
        adjusted_prob = tf.reshape(adjusted_prob, [-1, H * W, B, C])

        adjusted_net_out = tf.concat(
            [adjusted_coords_xy, adjusted_coords_wh, adjusted_euler, adjusted_c, adjusted_prob], 3)

        # 计算左上， 右下值
        wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])  # 比例值转化为具体指
        area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
        centers = coords[:, :, :, 0:2]  # x, y
        floor = centers - (wh * .5)     # 左上
        ceil = centers + (wh * .5)      # 右下

        # 计算 intersection area
        intersect_upleft = tf.maximum(floor, _upleft)
        intersect_botright = tf.minimum(ceil, _botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

        # calculate the best IOU, set 0.0 confidence for worse boxes
        iou = tf.truediv(intersect, _areas + area_pred - intersect)
        best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))      # 选取5个anchor中iou最大值
        best_box = tf.to_float(best_box)
        confs = tf.multiply(best_box, _confs)  # 选择最大的iou的anchor

        # class_loss
        # object_loss
        # noobject_loss
        # coord_loss
        # eular_loss

        sprob = cfg.CLASS_SCALE      # 1
        sconf = cfg.OBJECT_SCALE     # 5
        snoob = cfg.NOOBJECT_SCALE   # 1
        scoor = cfg.COORD_SCALE      # 1
        seuler = cfg.COORD_EULER_SCALE  # 1

        # TODO：loss 计算优化
        conid = snoob * (1. - confs) + sconf * confs                   # object_loss  noobject_loss   # (1, 512, 5)
        weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)     # (1, 512, 5, 4)
        cooid = scoor * weight_coo                                     # coord_loss  (1, 512, 5, 4)

        weight_eular = tf.concat(2 * [tf.expand_dims(confs, -1)], 3)   # (1, 512, 5, 2)
        eular = seuler * weight_eular                                  # (1, 512, 5, 2)

        weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)     # (1, 512, 5, 8)
        proid = sprob * weight_pro                                     # class_loss   (1, 512, 5, 8)

        fetch += [_probs, confs, conid, cooid, eular, proid]           # 应该还有 eular

        true = tf.concat([_coord, _euler, tf.expand_dims(confs, 3), _probs], 3)   # (1, 512, 5, 15)
        wght = tf.concat([cooid, eular, tf.expand_dims(conid, 3), proid], 3)      # (1, 512, 5, 15)
        print('Building {} loss'.format('darknet'))

        loss = tf.pow(adjusted_net_out - true, 2)
        #loss = smooth_l1(adjusted_net_out, true)
        loss = tf.multiply(loss, wght)
        loss = tf.reshape(loss, [-1, H * W * B * (6 + 1 + C)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss)
        tf.summary.scalar('{} loss'.format(''), loss)
        return loss

def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
        tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1

def loss(net_out, probs, confs, coord, euler,areas, upleft, botright, scope='loss'):
    '''
    net_out
        (None, CELL_SIZE_X, CELL_SIZE_Y, (6 + 5 + 1)*5)
        (None, CELL_SIZE_X, CELL_SIZE_Y, (6 + 8 + 1)*5)
    labels:
        # (None, N', 6(xyzlwhr))
        (None, N', 6(xyzlwhr))
    '''
    fetch = []
    with tf.variable_scope(scope):
        # 1 + 8 + 4 + 2 = 15
        W, H = cfg.CELL_SIZE_X, cfg.CELL_SIZE_Y  # 16 32
        B, C = cfg.NUM_ANCHORS, cfg.NUM_CLASSES   # 5 8

        _probs = probs
        _confs = confs
        _coord = coord
        _euler = euler
        _areas = areas
        _upleft = upleft
        _botright = botright

        net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (6 + 1 + C)])
        coords = net_out_reshape[..., :4]     # x y w h
        eulers = net_out_reshape[..., 4:6]    # im re
        anchors = cfg.ANCHORS

        coords = tf.reshape(coords, [-1, H * W, B, 4])
        adjusted_coords_xy = expit_tensor(coords[..., 0:2])  # sigma(x) y

        adjusted_coords_wh = tf.sqrt(
            tf.exp(coords[..., 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))  # w h
        adjusted_coords_wh = expit_tensor(adjusted_coords_wh)

        coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)   # x y w h

        adjusted_euler = tf.reshape(eulers, [-1, H * W, B, 2])            # im re

        adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 6])         # confidence  sigma(c)
        adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B, 1])

        adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 7:])    # category
        adjusted_prob = tf.reshape(adjusted_prob, [-1, H * W, B, C])

        adjusted_net_out = tf.concat(
            [adjusted_coords_xy, adjusted_coords_wh, adjusted_euler, adjusted_c, adjusted_prob], 3)

        # 计算左上， 右下值
        wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])  # 比例值转化为具体指
        area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
        centers = coords[:, :, :, 0:2]  # x, y
        floor = centers - (wh * .5)     # 左上
        ceil = centers + (wh * .5)      # 右下

        # 计算 intersection area
        intersect_upleft = tf.maximum(floor, _upleft)
        intersect_botright = tf.minimum(ceil, _botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

        # calculate the best IOU, set 0.0 confidence for worse boxes
        iou = tf.truediv(intersect, _areas + area_pred - intersect)
        best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))      # 选取5个anchor中iou最大值
        best_box = tf.to_float(best_box)
        confs = tf.multiply(best_box, _confs)                        # 选择最大的iou的anchor

        # class_loss
        # object_loss
        # noobject_loss
        # coord_loss
        # eular_loss

        sprob = cfg.CLASS_SCALE      # 1
        sconf = cfg.OBJECT_SCALE     # 5
        snoob = cfg.NOOBJECT_SCALE   # 1
        scoor = cfg.COORD_SCALE      # 1
        seuler = cfg.COORD_EULER_SCALE  # 1

        # TODO：loss 计算优化
        conid = snoob * (1. - confs) + sconf * confs                   # object_loss  noobject_loss   # (1, 512, 5)
        weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)     # (1, 512, 5, 4)
        cooid = scoor * weight_coo                                     # coord_loss  (1, 512, 5, 4)

        weight_eular = tf.concat(2 * [tf.expand_dims(confs, -1)], 3)   # (1, 512, 5, 2)
        eular = seuler * weight_eular                                  # (1, 512, 5, 2)

        weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)     # (1, 512, 5, 8)
        proid = sprob * weight_pro                                     # class_loss   (1, 512, 5, 8)

        fetch += [_probs, confs, conid, cooid, eular, proid]           # 应该还有 eular

        true = tf.concat([_coord, _euler, tf.expand_dims(confs, 3), _probs], 3)   # (1, 512, 5, 15)
        wght = tf.concat([cooid, eular, tf.expand_dims(conid, 3), proid], 3)      # (1, 512, 5, 15)
        print('Building {} loss'.format('darknet'))

        loss = tf.pow(adjusted_net_out[..., 6] - true[..., 6], 2)
        # loss = smooth_l1(adjusted_net_out, true)
        loss = tf.multiply(loss, wght[..., 6])
        loss = tf.reshape(loss, [-1, H * W * B * (1)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss)
        tf.summary.scalar('{} loss'.format('eular'), loss)
        return loss


if __name__ == '__main__':
    darknet = Darknet(8, 5, True, 1)



