#!/usr/bin/env python3
"""Task 3. Projection Block"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add

def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning for Image Recognition (2015).
    
    Parameters:
    A_prev: The output from the previous layer (tensor).
    filters: A tuple/list containing F11, F3, F12.
    s: The stride of the first convolution in both the main path and the shortcut connection.
    
    Returns:
    The activated output of the projection block.
    """
    F11, F3, F12 = filters
    
    # Main path
    # First 1x1 convolution (with stride s)
    X = Conv2D(F11, (1, 1), strides=(s, s), padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(seed=0))(A_prev)
    X = BatchNormalization(axis=3)(X)
    X = ReLU()(X)
    
    # Second 3x3 convolution
    X = Conv2D(F3, (3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = ReLU()(X)
    
    # Third 1x1 convolution
    X = Conv2D(F12, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    
    # Shortcut path
    shortcut = Conv2D(F12, (1, 1), strides=(s, s), padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(seed=0))(A_prev)
    shortcut = BatchNormalization(axis=3)(shortcut)
    
    # Add the main path and shortcut
    output = Add()([X, shortcut])
    output = ReLU()(output)
    
    return output

