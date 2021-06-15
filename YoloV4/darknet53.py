import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from backend import conv_bn_act, cross_stage_partial_block


def darknet53(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    X = conv_bn_act(inputs, filters=32, kernel_size=3, strides=1,
                    activation="mish")

    X = conv_bn_act(X, filters=64, kernel_size=3, strides=2, zero_pad=True,
                    padding="valid", activation="mish")

    route = conv_bn_act(X, filters=64, kernel_size=1, strides=1,
                        activation="mish")

    shortcut = conv_bn_act(X, filters=64, kernel_size=1, strides=1,
                           activation="mish")

    X = conv_bn_act(shortcut, filters=32, kernel_size=1, strides=1,
                    activation="mish")

    X = conv_bn_act(X, filters=64, kernel_size=3, strides=1, activation="mish")

    X = X + shortcut

    X = conv_bn_act(X, filters=64, kernel_size=1, strides=1, activation="mish")

    X = Concatenate()([X, route])

    X = conv_bn_act(X, filters=64, kernel_size=1, strides=1, activation="mish")

    X = cross_stage_partial_block(X, filters=128, num_blocks=2)

    output_256 = cross_stage_partial_block(X, filters=256, num_blocks=8)
    output_512 = cross_stage_partial_block(output_256, filters=512,
                                           num_blocks=8)
    output_1024 = cross_stage_partial_block(output_512, filters=1024,
                                            num_blocks=4)

    darknet = tf.keras.Model(inputs, [output_256, output_512, output_1024],
                             name='darknet')
    return darknet
