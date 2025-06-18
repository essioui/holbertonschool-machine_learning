#!/usr/bin/env python3
"""
Positional encoding for Transformer models.
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer
    Args:
        max_seq_len: maximum sequence length
        dm: model depth
    Returns:
        PE: numpy.ndarray of shape (max_seq_len, dm) containing
            the positional encoding vectors
    """
    # Create a matrix of shape (max_seq_len, dm)
    pos = np.arange(max_seq_len)[:, np.newaxis]

    # Create a matrix of shape (1, dm)
    i = np.arange(dm)[np.newaxis, :]

    # Calculate the positional encoding using sine and cosine functions
    angle_rate = 1 / np.power(10000, (2 * (i // 2)) / dm)

    # Calculate the angles for sine and cosine
    angle_rads = pos * angle_rate

    # Apply sine to even indices and cosine to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cosine to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads
