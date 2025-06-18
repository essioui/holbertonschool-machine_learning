#!/usr/bin/env python3
"""
RNN Decoder
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNN Decoder class
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        Args:
            vocab: size of the vocabulary
            embedding: dimension of the embedding vector
            units: number of units in the GRU cell
            batch: size of the batches
        """
        super(RNNDecoder, self).__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Call method for the RNNDecoder
        Args:
            x: input tensor of shape (batch, 1)
            s_prev: previous hidden state of shape (batch, units)
            hidden_states: shape (batch, input_seq_len, units)
        Returns:
            y: output tensor of shape (batch, vocab)
            s: new hidden state of shape (batch, units)
        """
        context, _ = self.attention(s_prev, hidden_states)

        x_embedded = self.embedding(x)

        x_embedded = tf.concat(
            [tf.expand_dims(context, 1), x_embedded], axis=-1
        )

        output, state = self.gru(x_embedded)

        output = tf.squeeze(output, axis=1)

        y = self.F(output)

        return y, state
