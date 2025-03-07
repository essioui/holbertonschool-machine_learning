#!/usr/bin/env python3
"""Task 3. Projection Block"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep
    Residual Learning for Image Recognition (2015).

    Arguments:
    A_prev -- output from the previous layer (tensor of shape (H, W, C))
    filters -- list or tuple containing F11, F3, F12:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1
        convolution and the shortcut connection
    s -- stride of the first convolution in both the main
    path and the shortcut connection (default is 2)

    Returns:
    activated_output -- the activated output of the
    projection block (tensor of shape (H/s, W/s, F12))
    """
    F11, F3, F12 = filters
    he_init = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(F11, (1, 1), strides=s, padding='valid',
                            kernel_initializer=he_init)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    conv2 = K.layers.Conv2D(F3, (3, 3), padding='same',
                            kernel_initializer=he_init)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    conv3 = K.layers.Conv2D(F12, (1, 1), padding='valid',
                            kernel_initializer=he_init)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    shortcut = K.layers.Conv2D(F12, (1, 1), strides=s, padding='valid',
                               kernel_initializer=he_init)(A_prev)
    shortcut_bn = K.layers.BatchNormalization(axis=3)(shortcut)

    add = K.layers.Add()([bn3, shortcut_bn])
    output = K.layers.Activation('relu')(add)

    return output
