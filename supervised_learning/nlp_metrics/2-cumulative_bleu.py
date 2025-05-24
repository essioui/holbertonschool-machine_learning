#!/usr/bin/env python3
"""
Cumulative N-gram BLEU score
"""
import numpy as np
from collections import Counter


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence
    Args:
        references is a list of reference translations
        sentence is a list containing the model proposed sentence
         is the size of the largest n-gram to use for evaluation
    Returns:
        the cumulative n-gram BLEU score
    """
    # Check if references is empty or sentence is empty
    def get_ngrams(words, n):
        """Helper to generate n-grams from a list of words"""
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

    # If references or sentence is empty, return 0.0
    precisions = []

    # If n is less than 1, return 0.0
    for i in range(1, n + 1):
        # Generate n-grams for the sentence
        sentence_ngrams = get_ngrams(sentence, i)
        sentence_counter = Counter(sentence_ngrams)

        max_ref_counts = {}
        # For each reference, count n-grams and keep the maximum counts
        for ref in references:
            ref_ngrams = get_ngrams(ref, i)
            ref_counter = Counter(ref_ngrams)
            for ngram in ref_counter:
                max_ref_counts[ngram] = max(
                    max_ref_counts.get(ngram, 0), ref_counter[ngram]
                )

        clipped_count = 0

        # Calculate clipped count for the sentence n-grams
        total_ngrams = sum(sentence_counter.values())

        # Iterate through the sentence n-grams and clip counts
        for ngram in sentence_counter:
            if ngram in max_ref_counts:
                clipped_count += min(
                    sentence_counter[ngram], max_ref_counts[ngram]
                )

        # Calculate precision for this n-gram size
        if total_ngrams == 0:
            precisions.append(0)
        else:
            precisions.append(clipped_count / total_ngrams)

    # If any precision is 0, BLEU = 0
    if any(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of log(precisions)
    log_precisions = [np.log(p) for p in precisions]
    mean_log_precision = sum(log_precisions) / n

    # Brevity Penalty
    c = len(sentence)
    ref_lens = [len(ref) for ref in references]
    r = min(ref_lens, key=lambda rl: abs(rl - c))

    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)

    return bp * np.exp(mean_log_precision)
