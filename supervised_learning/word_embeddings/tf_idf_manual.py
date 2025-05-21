#!/usr/bin/env python3
"""
TF-IDF
"""
import numpy as np
import re


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
    tokenized = [tokenize(sentence) for sentence in sentences]
    
    if vocab is None:
        vocab_set = set()
        for words in tokenized:
            vocab_set.update(words)
        features = sorted(vocab_set)
    else:
        features = vocab
    feature_idx = {word: i for i, word in enumerate(features)}
    s = len(sentences)
    f = len(features)
    
    # compute term frequency(TF)
    tf = np.zeros((s, f))   
    
    for i, words in enumerate(tokenized):
        world_count = len(words)
        for word in words:
            if word in feature_idx:
                tf[i, feature_idx[word]] += 1
        if world_count > 0:
            tf[i] /= world_count
            
    # compute inverse document frequency(IDF)
    df = np.zeros(f)
    for i, words in enumerate(tokenized):
        for word in words:
            if word in feature_idx:
                df[feature_idx[word]] += 1
    idf = np.log(s / (df + 1))
    
    # compute TF-IDF
    embeddings = tf * idf
    return embeddings, np.array(features)
    
    