import tensorflow as tf
import numpy as np
import cv2

from utils import conv_padding


def conv_block(inputs, filters, kernel_size, strides=1, training=True,
               data_format='channels_first'):
    inputs = conv_padding(inputs, filters=filters, kernel_size=kernel_size,
                          strides=strides, data_format=data_format)
    inputs = tf.keras.layers.BatchNormalization()(inputs, training=training)
    inputs = tf.keras.layers.LeakyReLU()(inputs)
    return inputs


def residual_conv_block(inputs, filters, training, data_format, strides=1):
    residual_connection = inputs

    inputs = conv_padding(inputs, filters, kernel_size=1, strides=strides,
                          data_format=data_format)
    inputs = tf.keras.layers.BatchNormalization()(inputs, training)
    inputs = tf.keras.layers.LeakyReLU()(inputs)

    inputs = conv_padding(inputs, filters * 2, kernel_size=3, strides=strides,
                          data_format=data_format)
    inputs = tf.keras.layers.BatchNormalization()(inputs, training)
    inputs = tf.keras.layers.LeakyReLU()(inputs)

    inputs += residual_connection

    return inputs


def darknet53(inputs, training, data_format):
    inputs = conv_block(inputs, filters=32, kernel_size=3, training=training,
                        data_format=data_format)
    inputs = conv_block(inputs, filters=64, kernel_size=3, strides=2,
                        training=training, data_format=data_format)

    inputs = residual_conv_block(inputs, filters=32, training=training,
                                 data_format=data_format)

    inputs = conv_block(inputs, filters=128, kernel_size=3, strides=2,
                        training=training, data_format=data_format)

    for repeat in range(2):
        inputs = residual_conv_block(inputs, filters=64, training=training,
                                     data_format=data_format)

    inputs = conv_block(inputs, filters=256, kernel_size=3, strides=2,
                        training=training, data_format=data_format)

    for repeat in range(8):
        inputs = residual_conv_block(inputs, filters=128, training=training,
                                     data_format=data_format)

    route_1 = inputs

    inputs = conv_block(inputs, filters=512, kernel_size=3, strides=2,
                        training=training, data_format=data_format)

    for repeat in range(8):
        inputs = residual_conv_block(inputs, filters=256, training=training,
                                     data_format=data_format)
    route_2 = inputs

    inputs = conv_block(inputs, filters=1024, kernel_size=3, strides=2,
                        training=training, data_format=data_format)

    for repeat in range(4):
        inputs = residual_conv_block(inputs, filters=512, training=training,
                                     data_format=data_format)
    route_3 = inputs

    return route_1, route_2, route_3
