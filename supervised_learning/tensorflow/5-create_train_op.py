#!/usr/bin/env python3
"""
This modules creates the training operation for the network
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution


def create_train_op(loss, alpha):
    """
    creates the training operation for the network
    Args:
        loss is the loss of the network’s prediction
        alpha is the learning rate
    Returns:
        an operation that trains the network using gradient descent
    """

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
