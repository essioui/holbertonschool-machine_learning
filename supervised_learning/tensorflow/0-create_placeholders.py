#!/usr/bin/env python3
"""
returns two placeholders, x and y, for the neural network
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Create placeholders
    Args:
        nx (_type_): _description_
        classes (_type_): _description_
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")
    return x, y
