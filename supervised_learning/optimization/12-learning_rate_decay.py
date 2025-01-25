#!/usr/bin/env python3
"""
Defines module  Learning Rate Decay Upgraded
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in tensorflow
    using inverse time decay
    Args:
        - alpha is the original learning rate
        - decay_rate is the weight used to determine
           the rate at which alpha will decay
        - decay_step is the number of passes of gradient descent
          that should occur before alpha is decayed further
    Returns:
        - the learning rate decay operation
    """
    updated_alpha_schedules = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return updated_alpha_schedules
