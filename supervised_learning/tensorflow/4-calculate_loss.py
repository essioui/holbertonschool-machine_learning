#!/usr/bin/env python3
"""
This modules calculates the softmax cross-entropy loss of a prediction
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution


def calculate_loss(y, y_pred):
    """
    calculates the softmax cross-entropy loss of a prediction
    Args:
        y is a placeholder for the labels of the input data
        y_pred is a tensor containing the network’s predictions
    Return:
        tensor containing the loss of the prediction
    """
    with tf.name_scope("softmax_cross_entropy_loss"):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2
            (labels=y, logits=y_pred),
            name="value"
        )
    return loss
