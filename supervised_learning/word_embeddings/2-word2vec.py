#!/usr/bin/env python3
"""
Train Word2Vec
"""
import gensim


def word2vec_model(
    sentences, vector_size=100, min_count=5, window=5, negative=5,
    cbow=True, epochs=5, seed=0, workers=1
):
    """
    Builds and trains a gensim word2vec model
    Args:
        sentences is a list of sentences to be trained on
        vector_size is the dimensionality of the embedding layer
        min_count is the minimum number of for use in training
        window is the maximum distance between the current and predicted word
        negative is the size of negative sampling
        cbow is a boolean True is for CBOW; False is for Skip-gram
        epochs is the number of iterations to train over
        seed is the seed for the random number generator
        workers is the number of worker threads to train the model
    Returns:
        the trained model
    """
    # choose the model type based on cbow
    # if cbow is True, use CBOW (sg=0), else use Skip-gram (sg=1)
    if cbow:
        sg = 0
    else:
        sg = 1

    # Create the Word2Vec model
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    # Build the vocabulary
    model.build_vocab(sentences)
    model.train(
        sentences, total_examples=model.corpus_count, epochs=model.epochs
    )

    return model
