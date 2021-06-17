import tensorflow as tf
import numpy as np
import cv2


def fixed_padding(inputs, kernel_size, data_format):
    padding_size = kernel_size - 1
    padding_start = padding_size // 2
    padding_end = padding_size - padding_start

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs,
                               [[0, 0], [0, 0], [padding_start, padding_end],
                                [padding_start, padding_end]])
    else:
        padded_inputs = tf.pad(inputs,
                               [[0, 0], [padding_start, padding_end],
                                [padding_start, padding_end], [0, 0]])

    return padded_inputs


def conv_padding(inputs, filters, kernel_size, data_format, strides=1):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    output = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                    strides=strides,
                                    padding=('SAME' if strides == 1
                                             else 'VALID'),
                                    use_bias=False, data_format=data_format
                                    )(inputs)
    return output


def non_maximum_suppression(inputs, num_classes, output_size, iou_threshold,
                            objectness_threshold):
    detections = tf.unstack(inputs)
    boxes_list = list()
    for detection in detections:
        detection = tf.boolean_mask(detections,
                                    detection[:, 4] > objectness_threshold)
        classes = tf.argmax(detection[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.cast(classes, dtype=tf.float32), axis=-1)
        boxes = tf.concat([detection[:, :5], classes], axis=-1)

        detection_dict = dict()

        for obj_class in range(num_classes):
            mask = tf.equal(boxes[:, 5], obj_class)
            mask_shape = mask.get_shape()
            if mask_shape.ndims != 0:
                class_boxes = tf.boolean_mask(boxes, mask)
                box_coords, box_scores, _ = tf.split(class_boxes, [4, 1, -1],
                                                     axis=-1)
                box_scores = tf.reshape(box_scores, [-1])
                indices = tf.image.non_max_suppression(box_coords, box_scores,
                                                       output_size,
                                                       iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                detection_dict[obj_class] = class_boxes[:, :5]
        boxes_list.append(detection_dict)
    return boxes_list


def build_boxes(inputs):
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    boxes = tf.concat([top_left_x, top_left_y,
                       bottom_right_x, bottom_right_y,
                       confidence, classes], axis=-1)

    return boxes


# Loads images in a 4D array
def load_images(img_names, model_size):
    imgs = []

    img = cv2.imread(img_names)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, model_size)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img[:, :, :3], axis=0)
    imgs.append(img)

    imgs = np.concatenate(imgs)

    return imgs


# Returns a list of class names read from `file_name`
def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names
