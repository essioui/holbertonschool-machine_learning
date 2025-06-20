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

    def __init__(self):
        """
        Initializes the Dataset object and loads the training and validation
        datasets. Also initializes tokenizers for Portuguese and English,
        and prepares the datasets by encoding the sentences.
        """
        # Load the Portuguese to English translation dataset
        data = tfds.load('ted_hrlr_translate/pt_to_en', as_supervised=True)
        self.data_train = data['train']
        self.data_valid = data['validation']

        # Initialize tokenizers using sentences from the training dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        # Apply encoding to both datasets
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset using pre-trained tokenizers and adapts them to
        the dataset.

        Args:
            data: tf.data.Dataset containing tuples of (pt, en) sentences.

        Returns:
        - tokenizer_pt: Trained tokenizer for Portuguese.
        - tokenizer_en: Trained tokenizer for English.
        """
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True,
            clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True,
            clean_up_tokenization_spaces=True)

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens, including start and end of sentence
        tokens.

        Args:
            pt: `tf.Tensor` containing the Portuguese sentence.
            en: `tf.Tensor` containing the corresponding English sentence.

        Returns:
        - pt_tokens: `np.ndarray` containing the Portuguese tokens.
        - en_tokens: `np.ndarray` containing the English tokens.
        """
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(pt_sentence,
                                             add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_sentence,
                                             add_special_tokens=False)

        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the `encode` method using `tf.py_function`.

        Args:
            pt: tf.Tensor of Portuguese sentence
            en: tf.Tensor of English sentence

        Returns:
        - pt_tokens: tf.Tensor of token ids (Portuguese)
        - en_tokens: tf.Tensor of token ids (English)
        """
        result_pt, result_en = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
