#!/usr/bin/env python3
"""
Extract Word2Vec model from Gensim and convert it to Keras Embedding layer.
"""
import tensorflow as tf



def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer
    Args:
        model: gensim word2vec model
    Returns:
        the trainable keras Embedding layer
    """
    # Get the weights of the model
    vocab_size = len(model.wv)

    # Get the embedding dimension
    embedding_dim = model.vector_size

    # Get the weights of the model
    weights = model.wv.vectors

    # Normalize the weights
    embadding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )

    return embadding_layer
