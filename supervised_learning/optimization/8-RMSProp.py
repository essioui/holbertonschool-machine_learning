#!/usr/bin/env python3
"""
Defines module RMSProp Upgraded
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow
    Args:
        alpha is the learning rate
        beta2 is the RMSProp weight (Discounting factor)
        epsilon is a small number to avoid division by zero
    Returns:
        optimizer
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
    return optimizer
