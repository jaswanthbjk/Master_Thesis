import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Lambda, MaxPool2D, \
    UpSampling2D

from backend import conv_bn_act, class_anchors_layer, bounding_box_regression, \
    non_maximum_suppression
from anchors import normalize_anchors, YOLOV4_ANCHORS
from darknet53 import darknet53
from tools import get_weights_by_keyword_or_path


def yolo_head(input_shapes, anchors, num_classes, training, yolo_max_boxes,
              yolo_iou_threshold, yolo_score_threshold):
    input_1 = Input(shape=filter(None, input_shapes[0]))
    input_2 = Input(shape=filter(None, input_shapes[1]))
    input_3 = Input(shape=filter(None, input_shapes[2]))

    X = conv_bn_act(input_1, filters=256, kernel_size=3, strides=1)
    output_1 = class_anchors_layer(X, num_anchors=len(anchors[0]),
                                   num_classes=num_classes)

    X = conv_bn_act(input_1, filters=256, kernel_size=3, strides=2,
                    zero_pad=True, padding="valid")

    X = Concatenate()([X, input_2])

    X = conv_bn_act(X, filters=256, kernel_size=1, strides=1)
    X = conv_bn_act(X, filters=512, kernel_size=3, strides=1)
    X = conv_bn_act(X, filters=256, kernel_size=1, strides=1)
    X = conv_bn_act(X, filters=512, kernel_size=3, strides=1)

    res_connection = conv_bn_act(X, filters=256, kernel_size=1, strides=1)
    X = conv_bn_act(res_connection, filters=512, kernel_size=3, strides=1)
    output_2 = class_anchors_layer(X, num_anchors=len(anchors[1]),
                                   num_classes=num_classes)

    X = conv_bn_act(res_connection, filters=512, kernel_size=3, strides=2,
                    zero_pad=True, padding="valid")
    X = Concatenate()([X, input_3])

    X = conv_bn_act(X, filters=512, kernel_size=1, strides=1)
    X = conv_bn_act(X, filters=1024, kernel_size=3, strides=1)
    X = conv_bn_act(X, filters=512, kernel_size=1, strides=1)
    X = conv_bn_act(X, filters=1024, kernel_size=3, strides=1)
    X = conv_bn_act(X, filters=512, kernel_size=1, strides=1)
    X = conv_bn_act(X, filters=1024, kernel_size=3, strides=1)

    output_3 = class_anchors_layer(X, num_anchors=len(anchors[2]),
                                   num_classes=num_classes)

    if training:
        head = Model([input_1, input_2, input_3],
                     [output_1, output_2, output_3], name="YOLO_HEAD")
        return head

    predictions_1 = Lambda(lambda X_input:
                           bounding_box_regression(X_input, anchors[0]))(output_1)
    predictions_2 = Lambda(lambda X_input:
                           bounding_box_regression(X_input, anchors[1]))(output_2)
    predictions_3 = Lambda(lambda X_input:
                           bounding_box_regression(X_input, anchors[2]))(output_3)

    output = Lambda(lambda x_input: non_maximum_suppression(
        x_input, max_boxes=yolo_max_boxes, iou_threshold=yolo_iou_threshold,
        prob_threshold=yolo_score_threshold), name="nms")([predictions_1,
                                                           predictions_2,
                                                           predictions_3])

    return tf.keras.Model([input_1, input_2, input_3], output, name="YOLO_HEAD")


def yolo_neck(input_shapes):
    input_1 = tf.keras.Input(shape=filter(None, input_shapes[0]))
    input_2 = tf.keras.Input(shape=filter(None, input_shapes[1]))
    input_3 = tf.keras.Input(shape=filter(None, input_shapes[2]))

    X = conv_bn_act(input_3, filters=512, kernel_size=1, strides=1)
    X = conv_bn_act(X, filters=1024, kernel_size=3, strides=1)
    X = conv_bn_act(X, filters=512, kernel_size=1, strides=1)

    maxpool_1 = MaxPool2D(pool_size=(5, 5), strides=1, padding="same")(X)
    maxpool_2 = MaxPool2D(pool_size=(9, 9), strides=1, padding="same")(X)
    maxpool_3 = MaxPool2D(pool_size=(13, 13), strides=1, padding="same")(X)

    spp = Concatenate()([maxpool_3, maxpool_2, maxpool_1, X])

    X = conv_bn_act(spp, filters=512, kernel_size=1, strides=1)
    X = conv_bn_act(X, filters=1024, kernel_size=3, strides=1)
    output_3 = conv_bn_act(X, filters=512, kernel_size=1, strides=1)
    X = conv_bn_act(output_3, filters=256, kernel_size=1, strides=1)

    intermediate_output = UpSampling2D()(X)

    X = conv_bn_act(input_2, filters=256, kernel_size=1, strides=1)
    X = Concatenate()([X, intermediate_output])

    X = conv_bn_act(X, filters=256, kernel_size=1, strides=1)
    X = conv_bn_act(X, filters=512, kernel_size=3, strides=1)
    X = conv_bn_act(X, filters=256, kernel_size=1, strides=1)
    X = conv_bn_act(X, filters=512, kernel_size=3, strides=1)
    output_2 = conv_bn_act(X, filters=256, kernel_size=1, strides=1)
    X = conv_bn_act(output_2, filters=128, kernel_size=1, strides=1)

    intermediate_output = tf.keras.layers.UpSampling2D()(X)

    X = conv_bn_act(input_1, filters=128, kernel_size=1, strides=1)
    X = tf.keras.layers.Concatenate()([X, intermediate_output])

    X = conv_bn_act(X, filters=128, kernel_size=1, strides=1)
    X = conv_bn_act(X, filters=256, kernel_size=3, strides=1)
    X = conv_bn_act(X, filters=128, kernel_size=1, strides=1)
    X = conv_bn_act(X, filters=256, kernel_size=3, strides=1)
    output_1 = conv_bn_act(X, filters=128, kernel_size=1, strides=1)

    yolo_neck = Model([input_1, input_2, input_3],
                      [output_1, output_2, output_3], name="YOLO_NECK")
    return yolo_neck


def yoloV4_model(input_shape, num_classes, anchors, training=False,
                 max_boxes=50, iou_threshold=0.5, prob_threshold=0.5,
                 weights="darknet"):
    feature_extractor = darknet53(input_shape)

    neck = yolo_neck(input_shapes=feature_extractor.output_shape)
    normalized_anchors = normalize_anchors(anchors, input_shape)

    head = yolo_head(neck.output_shape, normalized_anchors, num_classes,
                     training, max_boxes, iou_threshold, prob_threshold)

    input_image = Input(shape=input_shape)
    features_extracted_stage_1 = feature_extractor(input_image)
    features_extracted_stage_2 = neck(features_extracted_stage_1)
    features_extracted_stage_3 = head(features_extracted_stage_2)

    yolov4 = Model(inputs=input_image, outputs=features_extracted_stage_3)

    weights_path = get_weights_by_keyword_or_path(weights, model=yolov4)
    if weights_path is not None:
        yolov4.load_weights(str(weights_path), by_name=True, skip_mismatch=True)

    return yolov4


if __name__ == "__main__":
    yolov4_model = yoloV4_model((1024, 1024, 3), num_classes=7,
                                anchors=YOLOV4_ANCHORS)
    print(yolov4_model.summary())
