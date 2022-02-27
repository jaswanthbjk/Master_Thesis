import tensorflow as tf
import numpy as np
import cv2

from utils import conv_padding, build_boxes, non_maximum_suppression
from model_parameters import _ANCHORS as anchors


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


def Upsampling_layer(inputs, output_shape, data_format):
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = output_shape[3]
        new_width = output_shape[2]
    else:
        new_height = output_shape[2]
        new_width = output_shape[1]

    inputs = tf.image.resize(inputs, [new_height, new_width], method='nearest')

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs


def yolo_neck(inputs, filters, training, data_format):
    inputs = conv_block(inputs, filters=filters, kernel_size=1,
                        data_format=data_format, training=training)
    inputs = conv_block(inputs, filters=2 * filters, kernel_size=3,
                        data_format=data_format, training=training)

    inputs = conv_block(inputs, filters=filters, kernel_size=1,
                        data_format=data_format, training=training)
    inputs = conv_block(inputs, filters=2 * filters, kernel_size=3,
                        data_format=data_format, training=training)

    inputs = conv_block(inputs, filters=filters, kernel_size=1,
                        data_format=data_format, training=training)

    route = inputs

    inputs = conv_block(inputs, filters=2 * filters, kernel_size=3,
                        data_format=data_format, training=training)

    return route, inputs


def yolo_detection_branch(inputs, number_of_classes, anchors, image_size,
                          data_format):
    num_anchors = len(anchors)

    inputs = tf.keras.layers.Conv2D(
        filters=num_anchors * (5 + number_of_classes),
        kernel_size=1, strides=1, use_bias=True,
        data_format=data_format)(inputs)

    shape = inputs.get_shape().as_list()

    if data_format == 'channels_first':
        grid_shape = shape[2:4]
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    else:
        grid_shape = shape[1:3]

    inputs = tf.reshape(inputs, [-1,
                                 num_anchors * grid_shape[0] * grid_shape[1],
                                 5 + number_of_classes])
    strides = (image_size[0] // grid_shape[0], image_size[1] // grid_shape[1])

    box_centers, box_shapes, confidence, classes = \
        tf.split(inputs, [2, 2, 1, number_of_classes], axis=-1)

    x = tf.range(grid_shape[0], dtype=tf.float32)
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, num_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

    confidence = tf.nn.sigmoid(confidence)

    classes = tf.nn.sigmoid(classes)

    detections = tf.concat([box_centers, box_shapes, confidence, classes],
                           axis=-1)

    return detections


class YoloV3:
    def __init__(self, num_classes, input_size, output_size, iou_threshold,
                 objectness_threshold, data_format=None):
        if not data_format:
            if tf.test.is_built_with_cuda():
                self.data_format = 'channels_first'
            else:
                self.data_format = 'channels_last'
        self.num_classes = num_classes
        self.input_size = input_size
        self.output_size = output_size
        self.iou_threshold = iou_threshold
        self.objectness_threshold = objectness_threshold

    def __call__(self, images, training):
        with tf.compat.v1.variable_scope('yolov3_model'):
            if self.data_format == 'channels_first':
                images = tf.transpose(images, [0, 3, 1, 2])

            images = images / 255

            route_1, route_2, route_3 = darknet53(images, training,
                                                  data_format=self.data_format)

            route, inputs = yolo_neck(route_3, filters=512, training=training,
                                      data_format=self.data_format)

            detection_scale_1 = yolo_detection_branch(inputs, self.num_classes,
                                                      anchors[6:9],
                                                      self.input_size,
                                                      self.data_format)

            inputs = conv_block(inputs, filters=256, kernel_size=1,
                                training=training, data_format=self.data_format)

            upsample_size = route_2.get_shape().as_list()

            inputs = Upsampling_layer(inputs, output_shape=upsample_size,
                                      data_format=self.data_format)

            axis = 1 if self.data_format == 'channels_first' else 3

            inputs = tf.concat([inputs, route_2], axis=axis)

            route, inputs = yolo_neck(inputs, filters=256, training=training,
                                      data_format=self.data_format)

            detection_scale_2 = yolo_detection_branch(inputs, self.num_classes,
                                                      anchors[6:9],
                                                      self.input_size,
                                                      self.data_format)

            inputs = conv_block(route, filters=128, kernel_size=1,
                                training=training, data_format=self.data_format)

            upsample_size = route_1.get_shape().as_list()

            inputs = Upsampling_layer(inputs, output_shape=upsample_size,
                                      data_format=self.data_format)

            inputs = tf.concat([inputs, route_1], axis=axis)

            route, inputs = yolo_neck(inputs, filters=256, training=training,
                                      data_format=self.data_format)

            detection_scale_3 = yolo_detection_branch(inputs, self.num_classes,
                                                      anchors[6:9],
                                                      self.input_size,
                                                      self.data_format)

            inputs = tf.concat([detection_scale_1, detection_scale_2,
                                detection_scale_3], axis=1)

            inputs = build_boxes(inputs)

            boxes_dicts = non_maximum_suppression(inputs,
                                                  num_classes=self.num_classes,
                                                  output_size=self.output_size,
                                                  iou_threshold=
                                                  self.iou_threshold,
                                                  objectness_threshold=
                                                  self.objectness_threshold)

            return boxes_dicts
