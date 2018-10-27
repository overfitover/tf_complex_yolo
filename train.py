import tensorflow as tf
import os
from config import cfg
from utils import *
import argparse
from models.Model import Complex_YOLO
import time

parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=10, help='max epoch')
parser.add_argument('-t', '--tag', type=str, nargs='?', default='default', help='set log tag')
parser.add_argument('-b', '--batchsize', type=int, nargs='?', default=1, help='set batch size')
parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001, help='set learning rate')
parser.add_argument('-o', '--output-path', type=str, nargs='?', default='./predictions', help='result output dir')
args = parser.parse_args()

dataset_dir = cfg.DATA_DIR

train_dir = os.path.join(cfg.DATA_DIR, 'training')
val_dir = os.path.join(cfg.DATA_DIR, 'testing')

log_dir = os.path.join('./log', args.tag)
save_model_dir = os.path.join('./save_model', args.tag)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_model_dir, exist_ok=True)

def main():
    with tf.Graph().as_default():
        global save_model_dir
        start_epoch = 0
        global_counter = 0

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                                    visible_device_list=cfg.GPU_AVAILABLE,
                                    allow_growth=True)
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={
                "GPU": cfg.GPU_USE_COUNT,
            },
            allow_soft_placement=True,
        )

        with tf.Session(config=config) as sess:
            model = Complex_YOLO(
                batch_size=args.batchsize,
               learning_rate=args.lr
            )

            # param init
            # param init/restore
            # if tf.train.get_checkpoint_state(save_model_dir):
            #     print("Reading model parameters from %s" % save_model_dir)
            #     model.saver.restore(
            #         sess, tf.train.latest_checkpoint(save_model_dir))
            #     start_epoch = model.epoch.eval() + 1
            #     global_counter = model.global_step.eval() + 1
            # else:
            #     print("Created model with fresh parameters.")

            tf.global_variables_initializer().run()

            # training
            summary_interval = 10
            summary_val_interval = 10

            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            for epoch in range(start_epoch, args.max_epoch):
                counter = 0
                for batchdata in iterate_data(train_dir):   # 迭代数据进行训练

                    counter += 1
                    if counter % summary_interval == 0:
                        is_summary = True
                    else:
                        is_summary = False

                    global_counter += 1
                    ret = model.train_step(sess, batchdata, train=True, summary=is_summary)
                    print("train_loss: {}".format(ret[0]))

                    # print(counter, summary_interval, counter % summary_interval)
                    if counter % summary_interval == 0:
                        print("summary_interval now %d" % global_counter)
                        summary_writer.add_summary(ret[-1], global_counter)

                    # 模型保存
                    if global_counter % 100 == 0:
                        model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'),
                                         global_step=model.global_step)
                sess.run(model.epoch_add_op)


if __name__ == '__main__':
    main()