#!/usr/bin/env python3
"""
Create Masks for Transformer Models
"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    Create a padding mask for the input sequence.
    Args:
        seq (tf.Tensor): Input sequence tensor.
    Returns:
        tf.Tensor: Padding mask tensor.
    """
    # Create a mask where padding tokens (0) are set to 1
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Create a look-ahead mask for the decoder.
    Args:
        size (int): Size of the sequence.
    Returns:
        tf.Tensor: Look-ahead mask tensor.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """
    Create masks for the inputs and target sequences.
    Args:
        inputs (tf.Tensor): Input sequence tensor.
        target (tf.Tensor): Target sequence tensor.
    Returns:
        tuple: Tuple containing the encoder mask and the decoder mask.
    """
    # Create padding masks for the inputs and target sequences
    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(target)

    # Create look-ahead mask for the decoder
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    combined_mask = tf.maximum(decoder_mask, look_ahead_mask)

    return encoder_mask, combined_mask
