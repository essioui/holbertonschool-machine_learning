#!/usr/bin/env python3
"""
RNN Cell
"""
import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        Attributes:
            Wh: weights for the hidden state
            Wy: weights for the outputs
            bh: bias for the hidden state
            by: bias for the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the forward propagation of the cell
        Args:
            h_prev: previous hidden state of the shape (m, h)
            x_t: data input for the cell of shape (m, i)
                m: batch size
        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)
        # y = softmax(Wy * h_next + by)

        y_linear = np.dot(h_next, self.Wy) + self.by

        # Softmax activation
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y
