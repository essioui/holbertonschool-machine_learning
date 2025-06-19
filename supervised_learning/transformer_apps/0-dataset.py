#!/usr/bin/env python3
"""Dataset class for machine translation using BERT Fast tokenizers."""
import tensorflow_datasets as tfds
from transformers import BertTokenizerFast


class Dataset:
    """Loads and prepares dataset for machine translation."""

    def __init__(self):
        """Initialize the dataset and create tokenizers."""
        # upload the dataset from TensorFlow Datasets
        data, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                   as_supervised=True,
                                   with_info=True)
        self.data_train = data['train']
        self.data_valid = data['validation']

        # build the tokenizers for the dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset using BERT Fast tokenizers.

        Args:
            data: tf.data.Dataset in the format (pt, en)

        Returns:
            tokenizer_pt: BERT Fast tokenizer for Portuguese
            tokenizer_en: BERT Fast tokenizer for English
        """
        tokenizer_pt = BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = BertTokenizerFast.from_pretrained('bert-base-uncased')

        return tokenizer_pt, tokenizer_en
