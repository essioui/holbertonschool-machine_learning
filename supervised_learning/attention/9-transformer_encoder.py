#!/usr/bin/env python3
"""
Transformer Encoder
"""
import tensorflow as tf
import numpy as np

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Transformer Encoder
    """
    def __init__(
        self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1
    ):
        """
        Initialize the encoder
        Args:
            N: Number of blocks in the encoder
            dm: Dimensionality of the model
            h: Number of attention heads
            hidden: Dimensionality of the feed-forward network
            input_vocab: Size of the input vocabulary
            max_seq_len: Maximum sequence length
            drop_rate: Dropout rate
        """
        super(Encoder, self).__init__()

        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = (
            [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        )
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, padding_mask):
        """
        Forward pass of the encoder
        Args:
            x: Input tensor of shape (batch_size, input_seq_len)
            training: Boolean indicating whether the model is in training mode
            padding_mask: Mask for padding tokens
        Returns:
            Output tensor of shape (batch_size, input_seq_len, dm)
        """
        # Ensure x is a 2D tensor
        seq_len = tf.shape(x)[1]

        # Apply embedding and add positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        # Pass through each encoder block
        for block in self.blocks:
            x = block(x, training=training, mask=padding_mask)

        return x
