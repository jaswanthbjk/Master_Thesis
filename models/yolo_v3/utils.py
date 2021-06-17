import tensorflow as tf


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