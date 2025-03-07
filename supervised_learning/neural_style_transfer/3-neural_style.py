#!/usr/bin/env python3
"""
Module to define and initialize Neural Style Transfer (NST).
This module implements the NST process using a modified VGG19 model,
where MaxPooling layers are replaced with AveragePooling layers.
"""
import numpy as np
import tensorflow as tf
Model = tf.keras.models.Model
VGG19 = tf.keras.applications.VGG19


class NST:
    """
    Neural Style Transfer (NST) class.
    This class is responsible for performing NST using a VGG19-based model.
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
        Initializes an NST instance.

        Parameters:
        - style_image: The style reference image with shape (h, w, 3).
        - content_image: The content reference image with shape (h, w, 3).
        - alpha (float): Weight content loss. Must be a non-negative number.
        - beta (float): Weight style loss. Must be a non-negative number.
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
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so that its pixel values are between 0 and 1,
        and its largest side is 512 pixels while preserving the aspect ratio.

        Parameters:
        - image: The input image to be scaled with shape (h, w, 3).

        Returns:
        - tf.Tensor: A scaled image tensor with shape (1, new_h, new_w, 3).
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
        Loads the VGG19 model and replaces
        MaxPooling layers with AveragePooling.
        modified VGG19 model extracts features from
        both style and content layers.
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
                new_layer = layer.__class__.from_config(layer.get_config())
                new_layer.build(layer.input_shape)
                new_layer.set_weights(layer.get_weights())
                x = new_layer(x)

        modified_model = Model(inputs=inputs, outputs=x)

        outputs = ([modified_model.get_layer(name).output for name in
                    self.style_layers + [self.content_layer]])
        self.model = Model(inputs=modified_model.input,
                           outputs=outputs, name="model")

    @staticmethod
    def gram_matrix(input_layer):
        """
        Function calculate gram matrices
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")

        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape

        features = tf.reshape(input_layer, (h * w, c))

        gram = tf.matmul(features, features, transpose_a=True)

        gram /= tf.cast(h * w, tf.float32)

        return tf.expand_dims(gram, axis=0)

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost
        """
        nb_layers = len(self.style_layers)

        style_img = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        content_img = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style_img)
        content_outputs = self.model(content_img)

        style_features = [
            layer for layer in style_outputs[:nb_layers]
        ]
        self.content_feature = content_outputs[nb_layers:][0]

        self.gram_style_features = [
            NST.gram_matrix(layer) for layer in style_features
        ]
