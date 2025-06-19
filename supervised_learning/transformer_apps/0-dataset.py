#!/usr/bin/env python3
"""Dataset class for machine translation using BERT Fast tokenizers."""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Loads and prepares dataset for machine translation."""

    def __init__(self):
        """Initialize the dataset and create tokenizers."""
        # Load dataset from TensorFlow Datasets
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Build the tokenizers
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
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )
        return tokenizer_pt, tokenizer_en
