#!/usr/bin/env python3
"""
Self Attention
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Self Attention Layer
    """
    def __init__(self, units):
        """
        Initialize the layer with the number of units.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Compute the context vector and attention weights.
        """
        s_prev_expand = tf.expand_dims(s_prev, axis=1)

        score = self.V(
            tf.nn.tanh(self.W(s_prev_expand) + self.U(hidden_states))
        )

        weights = tf.nn.softmax(score, axis=1)

        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
