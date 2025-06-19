#!/usr/bin/env python3
"""Dataset class for machine translation using BERT Fast tokenizers."""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Loads and prepares dataset for machine translation."""

    def __init__(self):
        """Initialize the dataset and create tokenizers."""
        # Load train and validation splits separately
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Build the tokenizers (using train data only if you want, here just loading pretrained)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset()

    def tokenize_dataset(self):
        """
        Load pretrained BERT Fast tokenizers for Portuguese and English.

        Returns:
            tokenizer_pt: BERT Fast tokenizer for Portuguese
            tokenizer_en: BERT Fast tokenizer for English
        """
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased', use_fast=True
        )
        return tokenizer_pt, tokenizer_en