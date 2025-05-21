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
    if not sentences:
        return [], []

    vectorizer = TfidfVectorizer(vocabulary=vocab, lowercase=True)
    try:
        X = vectorizer.fit_transform(sentences)
    except ValueError:
        return [], []

    embeddings = X.toarray()
    features = vectorizer.get_feature_names_out().tolist()
    return embeddings, features
