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
    if vocab is None:
        vocab = sorted(
            set(word.lower() for s in sentences for word in s.split())
        )
    vectorizer = TfidfVectorizer(vocabulary=vocab, lowercase=True)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names_out().tolist()
    return embeddings, features
