#!/usr/bin/env python3
"""
Module define Initialize Neural Style Transfer
"""
import numpy as np
import tensorflow as tf
Model = tf.keras.models.Model
VGG19 = tf.keras.applications.VGG19


class NST:
    """
    Neural Style Transfer class
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
        ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes NST instance
        """
        if (not isinstance(style_image, np.ndarray) or
            len(style_image.shape) != 3 or
                style_image.shape[-1] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if (not isinstance(content_image, np.ndarray)
            or len(content_image.shape) != 3
                or content_image.shape[-1] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales image pixel values are between 0 and 1
        and its largest side is 512 pixels.
        """
        if (not isinstance(image, np.ndarray) or
            len(image.shape) != 3 or
                image.shape[-1] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape

        scale = 512 / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        resized_image = tf.image.resize(image, (new_h, new_w),
                                        method=tf.image.ResizeMethod.BICUBIC)

        scale_image = tf.cast(resized_image, tf.float32) / 255.0

        scale_image = tf.clip_by_value(scale_image, 0.0, 1.0)

        return tf.expand_dims(scale_image, axis=0)

    def load_model(self):
        """
        Loads the VGG19 model
        """
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        inputs = vgg.input
        x = inputs

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPool2D):

                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name.replace("max", "avg")
                )(x)
            else:
                x = layer.__class__.from_config(layer.get_config())(x)

        modified_model = Model(inputs=inputs, outputs=x)

        outputs = ([modified_model.get_layer(name).output for name in
                    self.style_layers + [self.content_layer]])
        self.model = Model(inputs=modified_model.input, outputs=outputs)
