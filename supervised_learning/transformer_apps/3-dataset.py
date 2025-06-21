#!/usr/bin/env python3
"""
Module that creates and prepares a dataset for machine translation
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    A class to load and prepare the TED HRLR translation dataset
    for machine translation from Portuguese to English.
    """

    def __init__(self, batch_size, max_len):
        """
        Initializes the Dataset object and prepares the data pipeline.

        Args:
            batch_size (int): The batch size for training/validation.
            max_len (int): Maximum number of tokens allowed per sentence.
        """
        self.batch_size = batch_size
        self.max_len = max_len

        # Load the dataset
        data = tfds.load('ted_hrlr_translate/pt_to_en', as_supervised=True)
        raw_train = data['train']
        raw_valid = data['validation']

        # Initialize tokenizers from training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(raw_train)

        # Prepare the training dataset
        self.data_train = raw_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(self.filter_max_length)
        self.data_train = self.data_train.map(
            lambda pt, en, pt_len, en_len: (pt, en)
        )
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(
            batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        # Prepare the validation dataset
        self.data_valid = raw_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(self.filter_max_length)
        self.data_valid = self.data_valid.map(
            lambda pt, en, pt_len, en_len: (pt, en)
        )
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None]))

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset using pre-trained tokenizers.

        Args:
            data: tf.data.Dataset of (pt, en) sentence pairs

        Returns:
            tokenizer_pt, tokenizer_en: trained tokenizers
        """
        pt_sentences, en_sentences = [], []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True)

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into token ids with special tokens.

        Returns:
            pt_tokens, en_tokens, pt_len, en_len
        """
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence, add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en_sentence, add_special_tokens=False
        )

        pt_len = len(pt_tokens)
        en_len = len(en_tokens)

        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens, pt_len, en_len

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for encode() using tf.py_function
        """
        pt_tokens, en_tokens, pt_len, en_len = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64, tf.int32, tf.int32]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        pt_len.set_shape(())
        en_len.set_shape(())
        return pt_tokens, en_tokens, pt_len, en_len

    def filter_max_length(self, pt, en, pt_len, en_len):
        """
        Filter out sentence pairs where either exceeds max_len
        """
        return tf.logical_and(pt_len <= self.max_len, en_len <= self.max_len)

    @property
    def pt_tokenizer(self):
        """Property to access Portuguese tokenizer"""
        return self.tokenizer_pt

    @property
    def en_tokenizer(self):
        """Property to access English tokenizer"""
        return self.tokenizer_en