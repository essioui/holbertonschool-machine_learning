#!/usr/bin/env python3
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf

# importing necessary modules
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def translate_sentence(transformer, dataset, sentence, max_len):
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
    print("please wait..")

    # upload the dataset
    dataset = Dataset(batch_size=1, max_len=40)
    input_vocab_size = dataset.tokenizer_pt.vocab_size + 2
    target_vocab_size = dataset.tokenizer_en.vocab_size + 2

    # build the transformer model
    transformer = Transformer(4, 128, 8, 512,
                              input_vocab_size, target_vocab_size,
                              50, 50)

    # build the model and compile it
    dummy_input = tf.constant([[1, 2, 3]])
    dummy_target = tf.constant([[1, 2]])
    enc_mask, combined_mask, dec_mask = create_masks(dummy_input, dummy_target)
    _ = transformer(dummy_input, dummy_target,
                    training=False,
                    encoder_mask=enc_mask,
                    look_ahead_mask=combined_mask,
                    decoder_mask=dec_mask)

    # load the trained weights
    transformer.load_weights("transformer.weights.h5")
    print("uploaded the trained weights successfully.\n")

    print("input text with portugal language \n")
    print("wtite quite for exit\n")

    while True:
        sentence = input("📝 > ").strip()
        if sentence.lower() in {"exit", "quit"}:
            print("quit the program")
            break

        translation = translate_sentence(transformer, dataset, sentence, max_len=40)
        print("translate:", translation, "\n")
