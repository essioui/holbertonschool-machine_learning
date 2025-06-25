#!/usr/bin/env python3
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf

# importing necessary modules
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer
train_transformer = __import__('5-train').train_transformer


def translate_sentence(transformer, dataset, sentence, max_len):
    """
    Translate a sentence from Portuguese to English using the trained Transformer model.
    """
    input_ids = dataset.tokenizer_pt.encode(sentence)
    input_ids = input_ids[:max_len]
    encoder_input = tf.expand_dims(input_ids, 0)

    start_token = dataset.tokenizer_en.vocab_size
    end_token = dataset.tokenizer_en.vocab_size + 1
    output = tf.expand_dims([start_token], 0)

    for _ in range(max_len):
        enc_mask, combined_mask, dec_mask = create_masks(encoder_input, output)

        predictions = transformer(encoder_input, output,
                                  training=False,
                                  encoder_mask=enc_mask,
                                  look_ahead_mask=combined_mask,
                                  decoder_mask=dec_mask)

        predicted_id = tf.argmax(predictions[:, -1, :], axis=-1).numpy()[0]

        if predicted_id == end_token:
            break

        output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)

    return dataset.tokenizer_en.decode(output[0].numpy())


if __name__ == "__main__":
    # load the dataset
    dataset = Dataset(batch_size=1, max_len=40)

    # apply the training function to train the Transformer model
    transformer = train_transformer(4, 128, 8, 512, 32, 40, 2)

    # text to translate
    sentence = "testei o meu modelo depois de o treinar"
    translated = translate_sentence(transformer, dataset, sentence, max_len=40)

    print("original text", sentence)
    print("translate:", translated)
    
    transformer.save_weights("transformer.weights.h5")
