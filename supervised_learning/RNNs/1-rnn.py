#!/usr/bin/env python3
"""
RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.
    Args:
        rnn_cell: an instance of RNNCell
        X: data to be used, shape (t, m, i)
            t: number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state, shape (m, h)
            h: dimensionality of the hidden state
    Returns:
        H: all hidden states, shape (t, m, h)
            h: dimensionality of the hidden state
        Y: outputs, shape (t, m, o)
            o: dimensionality of the output
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[0]

    H = np.zeros((t + 1, m, h))

    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        # h_prev is the previous hidden state
        h_prev = H[step]

        # x_t is the current data point
        x_t = X[step]

        # h_next is the next hidden state
        h_next, y = rnn_cell.forward(h_prev, x_t)

        # y is the output
        H[step + 1] = h_next

        if step == 0:
            o = y.shape[1]
            Y = np.zeros((t, m, o))

        # Y[step] is the output at time step step
        Y[step] = y

    # Return the hidden states and outputs
    return H, Y
