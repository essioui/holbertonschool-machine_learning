#!/usr/bin/env python3
"""
N-gram BLEU score
"""
from collections import Counter
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence
    Args:
        references is a list of reference translations
        sentence is a list containing the model proposed sentence
        n is the size of the n-gram to use for evaluation
    Returns:
        the n-gram BLEU score
    """
    # Check if references is empty or sentence is empty
    def get_ngrams(words, n):
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

    sentence_ngrams = get_ngrams(sentence, n)
    sentence_counter = Counter(sentence_ngrams)

    # calculate maximum counts for each n-gram in references
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        ref_counter = Counter(ref_ngrams)
        for ngram in ref_counter:
            max_ref_counts[ngram] = max(
                max_ref_counts.get(ngram, 0), ref_counter[ngram]
            )

    # calculate clipped counts
    clipped_count = 0
    for ngram in sentence_counter:
        if ngram in max_ref_counts:
            clipped_count += min(
                sentence_counter[ngram], max_ref_counts[ngram]
            )

    # if no n-grams are found, return 0.0
    total_ngrams = sum(sentence_counter.values())
    if total_ngrams == 0:
        return 0.0

    # calculate precision
    precision = clipped_count / total_ngrams

    # calculate brevity penalty
    c = len(sentence)
    ref_lens = [len(ref) for ref in references]
    r = min(ref_lens, key=lambda rl: abs(rl - c))

    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)

    return bp * precision
