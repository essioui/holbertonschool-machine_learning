#!/usr/bin/env python3
"""
gated recurrent unit GRU Cell
"""
import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function
    Args:
        x: input value
    Returns:
        Sigmoid of x
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Softmax activation function
    Args:
        x: input value
    Returns:
        Softmax of x
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


class GRUCell:
    """
    Represents a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
        The public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
            Wz and bz are for the update gate
            Wr and br are for the reset gate
            Wh and bh are for the intermediate hidden state
            Wy and by are for the output
        """
        self.Wz = np.random.randn(i, h)
        self.Uz = np.random.randn(h, h)
        self.bz = np.zeros((1, h))

        self.Wr = np.random.randn(i, h)
        self.Ur = np.random.randn(h, h)
        self.br = np.zeros((1, h))

        self.Wh = np.random.randn(i, h)
        self.Uh = np.random.randn(h, h)
        self.bh = np.zeros((1, h))

        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step
        Argrs:
            h_prev is the previous hidden state
            x_t is the input data for the cell
        Returns:
            h_next is the next hidden state
            y_t is the output of the cell
        """
        z = sigmoid(np.dot(x_t, self.Wz) + np.dot(h_prev, self.Uz) + self.bz)

        r = sigmoid(np.dot(x_t, self.Wr) + np.dot(h_prev, self.Ur) + self.br)

        h_tilde = np.tanh(np.dot(x_t, self.Wh) + np.dot(
            r * h_prev, self.Uh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde

        y = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
