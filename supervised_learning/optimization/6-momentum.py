#!/usr/bin/env python3
"""
Defines module Momentum Upgraded
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
     Sets up the gradient descent with momentum
     optimization algorithm in TensorFlow
     Args:
        alpha is the learning rate
        beta1 is the momentum weigh
    Returns:
        optimizer
    """
    optimizer = tf.keras.optimizers.SGD(alpha, beta1)
    return optimizer
