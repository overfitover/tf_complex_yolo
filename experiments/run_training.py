import argparse
import os

import tensorflow as tf


tf.logging.set_verbosity(tf.logging.ERROR)


def train(model_config, train_config, dataset_config):
    pass

    # dataset =
    # train_val_test = 'train'
    #
    # with tf.Graph().as_default():
    #     if model_name == 'rpn_model':
    #         model = Complex_YOLO(model_config,
    #                             train_val_test=train_val_test,
    #                             dataset = dataset)
    #
    #     trainer.train(model, train_config)

def main(_):
    parser = argparse.ArgumentParser()

    # parser.add_argument('')
    args = parser.parse_args()

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    train()


if __name__ == '__main__':
    tf.app.run()