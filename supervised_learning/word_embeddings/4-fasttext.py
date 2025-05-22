#!/usr/bin/env python3
"""
FastText
"""
import gensim


def fasttext_model(
    sentences, vector_size=100, min_count=5, negative=5,
    window=5, cbow=True, epochs=5, seed=0, workers=1
):
    """
    Builds and trains a genism fastText model
    Args:
        sentences is a list of sentences to be trained on
        vector_size is the dimensionality of the embedding layer
        min_count is the min num of occurrences of a word for use in training
        window the max distance between the current and predicted word
        negative is the size of negative sampling
        cbow if True is for CBOW; False is for Skip-gram
        epochs is the number of iterations to train over
        seed is the seed for the random number generator
        workers is the number of worker threads to train the model
    Returns:
        the trained model
    """
    if cbow:
        sg = 0
    else:
        sg = 1

    # Create the FastText model
    model = gensim.models.FastText(
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
