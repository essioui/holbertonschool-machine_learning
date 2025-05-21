#!/usr/bin/env python3
"""
Bag of words" embedding
"""
import numpy as np
import re
from typing import List, Tuple


def tokenize(sentence):
    """Tokenizes a sentence into words
    Args:
        sentence (str): the sentence to tokenize
    Returns:
        list of words in the sentence
    """
    sentence = sentence.lower()
    sentence = re.sub(r"'s", "", sentence)
    sentence = re.sub(r"[^a-zA-Z\s]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence.split()


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix
    Args:
        sentences (list): list of sentences
        vocab (list): list of words to include in the vocabulary
    Returns:
        embeddings, features
            embeddings is a numpy.ndarray of shape (s, f)
                s is the number of sentences in sentences
                f is the number of features analyzed
            features is a list of the features used for embedding
    """
    tokenized = [tokenize(sentence) for sentence in sentences]

    if vocab is None:
        vocab_set = set()
        for words in tokenized:
            vocab_set.update(words)
        features = sorted(vocab_set)
    else:
        features = vocab

    feature_idx = {word: i for i, word in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, words in enumerate(tokenized):
        for word in words:
            if word in feature_idx:
                embeddings[i, feature_idx[word]] += 1

    return embeddings, np.array(features)
