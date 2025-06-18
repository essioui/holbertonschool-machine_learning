#!/usr/bin/env python3
"""
Transformer Encoder
"""
import tensorflow as tf

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
        Initializes the Encoder.
        Args:
            N (int): Number of blocks in the encoder.
            dm (int): Dimensionality of the model.
            h (int): Number of heads in the multi-head attention.
            hidden (int): Dimensionality of the feed-forward network.
            input_vocab (int): Size of the input vocabulary.
            max_seq_len (int): Maximum sequence length for positional encoding.
            drop_rate (float): Dropout rate.
        Returns:
            None
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
        Forward pass of the Encoder.
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_len).
            training (bool): Whether the model is in training mode.
            padding_mask (tf.Tensor): Padding mask tensor.
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, seq_len, dm).
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training, mask=padding_mask)

        return x
