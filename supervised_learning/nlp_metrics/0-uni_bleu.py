#!/usr/bin/env python3
"""
Unigram BLEU score
"""
from collections import Counter
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    Args:
        references is a list of reference translations
        sentence is a list containing the model proposed sentence
    Returns:
        the unigram BLEU score
    """
    # Check if references is empty or sentence is empty
    sentence_counter = Counter(sentence)

    # calculate maximum counts for each word in references
    max_counts = {}
    for ref in references:
        ref_counter = Counter(ref)
        for word, count in ref_counter.items():
            max_counts[word] = max(max_counts.get(word, 0), count)

    # calculate clipped counts
    clipped_counts = 0
    for word in sentence_counter:
        if word in max_counts:
            clipped_counts += min(sentence_counter[word], max_counts[word])

    # Calculate total counts of words in the sentence
    total_counts = sum(sentence_counter.values())

    # If total counts is zero, return 0.0 to avoid division by zero
    if total_counts == 0:
        return 0.0

    # Calculate precision
    precision = clipped_counts / total_counts

    # Calculate brevity penalty
    c = len(sentence)
    r_len = min(len(ref) for ref in references)

    if c > r_len:
        bp = 1.0
    else:
        bp = np.exp(1 - r_len / c)

    return bp * precision
