import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, \
    ZeroPadding2D, Concatenate, Reshape


def conv_bn_act(inputs, filters, kernel_size, strides, padding="same",
                zero_pad=False, activation="leaky_relu"):
    if zero_pad:
        inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)

    X = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding=padding, use_bias=False)(inputs)

    X = BatchNormalization()(X)

    if activation == "leaky_relu":
        X = LeakyReLU(alpha=0.1)(X)
    elif activation == "mish":
        X = tfa.activations.mish(X)

    return X


def residual_block(inputs, num_blocks):
    _, _, _, filters = inputs.shape

    X = inputs
    for _ in range(num_blocks):
        block_input = X
        X = conv_bn_act(X, filters=filters, kernel_size=1, strides=1,
                        activation="mish")
        X = conv_bn_act(X, filters=filters, kernel_size=3, strides=1,
                        activation="mish")
        X = X + block_input
    return X


def cross_stage_partial_block(inputs, filters, num_blocks):
    half_filters = filters // 2

    X = conv_bn_act(inputs, filters, kernel_size=3, strides=2, zero_pad=True,
                    padding="valid", activation="mish")
    route = conv_bn_act(X, filters=half_filters, kernel_size=1, strides=1,
                        activation="mish")
    X = conv_bn_act(X, filters=half_filters, kernel_size=1, strides=1,
                    activation="mish")
    X = residual_block(X, num_blocks=num_blocks)
    X = conv_bn_act(X, filters=half_filters, kernel_size=1, strides=1,
                    activation="mish")
    X = Concatenate()([X, route])

    X = conv_bn_act(X, filters, kernel_size=1, strides=1, activation="mish")

    return X


def class_anchors_layer(inputs, num_anchors, num_classes):
    X = Conv2D(filters=num_anchors * (num_classes + 5), kernel_size=1,
               strides=1,
               padding="same", use_bias=True)(inputs)
    X = Reshape((X.shape[1], X.shape[2], num_anchors, num_classes + 5))(X)
    return X


def bounding_box_regression(features_per_stage, anchors_per_stage):
    grid_size_x, grid_size_y = features_per_stage.shape[1], \
                               features_per_stage.shape[2]
    num_classes = features_per_stage[-1] - 5

    box_xy, box_wh, objectess_score, prob = tf.split(features_per_stage,
                                                     (2, 2, 1, num_classes),
                                                     axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectess_score = tf.sigmoid(objectess_score)
    prob = tf.sigmoid(prob)

    grid = tf.meshgrid(tf.range(grid_size_y), tf.range(grid_size_x))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.constant(
        [grid_size_y, grid_size_x], dtype=tf.float32)
    box_wh = tf.exp(box_wh) * anchors_per_stage

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bounding_box = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bounding_box, objectess_score, prob


def non_maximum_suppression(features, max_boxes, iou_threshold, prob_threshold):
    box_per_stage, objectness_per_stage, prob_per_stage = [], [], []
    for feature in features:
        num_boxes = feature[0].shape[1] * feature[0].shape[2] * \
                    feature[0].shape[3]
        box_per_stage.append(tf.reshape(feature[0],
                                        (tf.shape(feature[0])[0], num_boxes,
                                         feature[0].shape[-1])))
        objectness_per_stage.append(tf.reshape(feature[1],
                                               (tf.shape(feature[1])[0],
                                                num_boxes,
                                                feature[1].shape[-1])))
        prob_per_stage.append(tf.reshape(feature[2],
                                         (tf.shape(feature[2])[0],
                                          num_boxes, feature[2].shape[-1])))

        bounding_box = tf.concat(box_per_stage, axis=1)
        objectness = tf.concat(objectness_per_stage, axis=1)
        class_probs = tf.concat(prob_per_stage, axis=1)

        boxes, scores, classes, valid_detections = \
            tf.image.combined_non_max_suppression(
                boxes=tf.expand_dims(bounding_box, axis=2),
                scores=objectness * class_probs,
                max_output_size_per_class=max_boxes,
                max_total_size=max_boxes,
                iou_threshold=iou_threshold,
                score_threshold=prob_threshold)
        return [boxes, scores, classes, valid_detections]

