#!/usr/bin/env python3
"""
TF_IDF
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
     Creates a TF-IDF embedding
     Args:
        sentences is a list of sentences to analyze
        vocab is a list of the vocabulary words to use for the analysis
    Returns:
        embeddings, features
            s is the number of sentences in sentences
            f is the number of features analyzed
        features is a list of the features used for embeddings
    """
    # Initialize the TF-IDF vectorizer with the given vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform the sentences to get the TF-IDF embeddings
    embeddings = vectorizer.fit_transform(sentences)

    # Extract the features (words) used by the vectorizer
    features = vectorizer.get_feature_names_out()

    return embeddings.toarray(), features

