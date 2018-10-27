import tensorflow as tf
from config import cfg
from models.darknet import Darknet
from utils import *

class Complex_YOLO(object):

    def __init__(self,
                batch_size=1,
                learning_rate=0.001,
                max_gradient_norm =5.0,
                avail_gpus = ['0']):

        # hyper parameters and status
        self.single_batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(1, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.avail_gpus = avail_gpus
        self.is_train = tf.placeholder(tf.bool, name='phase')

        self.classes = cfg.NUM_CLASSES
        self.num_anchors = cfg.NUM_ANCHORS
        
        # For the first epochs, we started with a small learning rate to prevent from diversion.
        # After some epochs, we scaled the learning rate up and continued to gradually decrease
        # it for up to 1,000 epochs

        boundaries = [80, 120]
        values = [self.learning_rate, self.learning_rate * 0.1, self.learning_rate * 0.01]
        lr = tf.train.piecewise_constant(self.epoch, boundaries, values)

        # build graph
        # input placeholders
        self.is_train = tf.placeholder(tf.bool, name='phase')
        self.voxel_feature = []
        self.targets = []
        self.erpn_output = []
        self.opt = tf.train.AdamOptimizer(lr)
        self.tower_grads = []

        with tf.variable_scope(tf.get_variable_scope()):
            for idx, dev in enumerate(self.avail_gpus):
                with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)):    
                    # (self, classes, num_anchors, is_training, center=True, batch_size)

                    darknet = Darknet(self.classes, self.num_anchors, self.is_train, self.single_batch_size)
                    tf.get_variable_scope().reuse_variables()

                    # input output
                    self.net_output = darknet.output
                    self.voxel_feature.append(darknet.input)

                    # loss and grad
                    if idx == 0:
                        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    self.loss = darknet.loss

                    self.params = tf.trainable_variables()
                    gradients = tf.gradients(self.loss, self.params)
                    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
                        gradients, max_gradient_norm)

                    # self.yolo_loss = darknet.yolo_loss
                    # self.erpn_loss = darknet.erpn_loss
                    self.tower_grads.append(clipped_gradients)

        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # loss and optimizer
        
        with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.grads = self.average_gradients(self.tower_grads)
            self.update = [self.opt.apply_gradients(
                zip(self.grads, self.params), global_step=self.global_step)]
            self.gradient_norm = tf.group(*self.gradient_norm)

        self.update.extend(self.extra_update_ops)
        self.update = tf.group(*self.update)

        self.train_bv = tf.placeholder(tf.uint8, [None, cfg.BV_LOG_FACTOR * cfg.INPUT_HEIGHT, cfg.BV_LOG_FACTOR*cfg.INPUT_WIDTH, 3])
        self.train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', self.loss),
            # tf.summary.scalar('train/yolo_loss', self.yolo_loss),
            # tf.summary.scalar('train/erpn_loss', self.erpn_loss),
            tf.summary.image('train/bird_view_lidar', self.train_bv),
            *[tf.summary.histogram(each.name, each) for each in self.vars + self.params]
        ])

    def train_step(self, session, data, train=False):
        # data 
        # voxel n * m * 3
        #    np.array(tag),
        #    np.array(labels),
        #    np.array(voxel),
        #    np.array(rgb),
        #    np.array(raw_lidar)
        tag = data[0]
        label = data[1]
        voxel_feature = data[2]
        lidar = data[4]
        # targets = cal_target(label, )
        input_feed = {}
        input_feed[self.is_train] = True
        for idx in range(len(self.avail_gpus)):
            input_feed[self.voxel_feature[idx]] = voxel_feature[idx]
            #input_feed[self.targets[idx]] = targets[idx * self.single_batch_size: (idx + 1) * self.single_batch_size]


        if train:
            output_feed = [self.loss, self.update]
        else:
            output_feed = [self.loss]

        return session.run(output_feed, input_feed)

    def predict_step(self):
        pass

    def validate_step(self):
        pass

    def average_gradients(self, tower_grads):
        # ref:
        # https://github.com/tensorflow/models/blob/6db9f0282e2ab12795628de6200670892a8ad6ba/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L103
        # but only contains grads, no vars
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            grad_and_var = grad
            average_grads.append(grad_and_var)
        return average_grads


if __name__ == '__main__':

    train_dir = '/home/yxk/project/data/training'
    with tf.Session() as sess:
        model = Complex_YOLO(
            batch_size=1,
            learning_rate=0.001
        )
        tf.global_variables_initializer().run()
        # training
        for epoch in range(0, 150):
            counter = 0
            for batchdata in iterate_data(train_dir):  # 迭代数据
                counter += 1
                ret = model.train_step(sess, batchdata, train=True)
            # sess.run(model.epoch_add_op)
