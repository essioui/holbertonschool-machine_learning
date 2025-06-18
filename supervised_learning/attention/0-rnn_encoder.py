#!/usr/bin/env python3
"""
RNN Encoder
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNN Encoder class
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the RNN Encoder
        Args:
            vocab: size of the vocabulary
            embedding: dimensionality of the embedding vector
            units: number of units in the RNN cell
            batch: batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Initialize the hidden state
        Returns:
            Tensor of shape (batch, units) filled with zeros
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Forward pass of the RNN Encoder
        Args:
            x: input tensor of shape (batch, sequence_length)
            initial: initial hidden state
        Returns:
            outputs: output tensor of shape (batch, sequence_length, units)
            hidden: final hidden state of shape (batch, units)
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
