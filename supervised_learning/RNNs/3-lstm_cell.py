#!/usr/bin/env python3
"""
Long Short-Term Memory (LSTM)
"""
import numpy as np


class LSTMCell:
    """
    Represents an LSTM unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        Attributes:
            Wf and bf are for the forget gate
            Wu and bu are for the update gate
            Wc and bc are for the intermediate cell state
            Wo and bo are for the output gate
            Wy and by are for the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Calculates the forward propagation of the cell
        Args:
            h_prev: previous hidden state of the shape (m, h)
            c_prev: previous cell state of the shape (m, h)
            x_t: data input for the cell of shape (m, i)
                m: batch size
        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        ft = self.sigmoid(np.dot(concat, self.Wf) + self.bf)
        ut = self.sigmoid(np.dot(concat, self.Wu) + self.bu)
        ct = np.tanh(np.dot(concat, self.Wc) + self.bc)
        c_next = ft * c_prev + ut * ct
        ot = self.sigmoid(np.dot(concat, self.Wo) + self.bo)
        h_next = ot * np.tanh(c_next)

        y_linear = np.dot(h_next, self.Wy) + self.by

        # Softmax activation
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, c_next, y

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        Args:
            x is the input data
        Returns:
            The sigmoid of x
        """
        return 1 / (1 + np.exp(-x))
