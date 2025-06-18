#!/usr/bin/env python3
"""
Multi Head Attention
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi Head Attention Class
    """
    def __init__(self, dm, h):
        """
        Initialize the class
        Args:
            dm: the dimensionality of the model
            h: the number of heads
        """
        super(MultiHeadAttention, self).__init__()

        # Validate inputs
        if dm % h != 0:
            raise ValueError("dm must be divisible by h")

        self.dm = dm
        self.h = h
        self.depth = dm // h

        # Define the weight matrices for queries, keys, and values
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        # Define the linear layer for output
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth)
        Transpose the result to shape (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask=None):
        """
        Call method for Multi Head Attention
        Args:
            Q: query matrix
            K: key matrix
            V: value matrix
            mask: mask matrix
        Return:
            output and attention weights
        """
        batch_size = tf.shape(Q)[0]

        # Linear projections
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled Dot-Product Attentio
        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        # Concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dm)
        )

        # Final linear layer
        output = self.linear(concat_attention)

        return output, weights
