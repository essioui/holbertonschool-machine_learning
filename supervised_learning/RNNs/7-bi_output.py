#!/usr/bin/env python3
"""
Bidirectional Cell Forward
"""
import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden states
            o is the dimensionality of the outputs
        Attributes:
            Whf and bhf are for the hidden states in the forward direction
            Whb and bhb are for the hidden states in the backward direction
            Wy and by are for the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction
        Args:
            x_t is a numpy.ndarray of shape (m, i)
                m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h)
        Returns:
            h_next: the next hidden state
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(
            np.matmul(concat, self.Whf) + self.bhf
        )

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction
        Args:
            x_t is a numpy.ndarray of shape (m, i)
                m is the batch size for the data
            h_next is a numpy.ndarray of shape (m, h)
        Returns:
            h_pev: the previous hidden state
        """
        concat = np.concatenate((h_next, x_t), axis=1)

        h_prev = np.tanh(
            np.matmul(concat, self.Whb) + self.bhb
        )

        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN
        Args:
            H is a numpy.ndarray of shape (t, m, 2 * h)
                t is the number of time steps
                m is the batch size for the data
                h is the dimensionality of the hidden states
        Returns:
            Y: the outputs
        """
        t, m, _ = H.shape

        # Concatenate the forward and backward hidden states
        y_linear = np.matmul(H, self.Wy) + self.by

        # Apply softmax on last axis
        e = np.exp(y_linear - np.max(y_linear, axis=2, keepdims=True))

        Y = e / np.sum(e, axis=2, keepdims=True)

        return Y
