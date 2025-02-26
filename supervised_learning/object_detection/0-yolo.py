#!/usr/bin/env python3
"""
Module define Initialize Yolo
"""
from tensorflow import keras as k


class Yolo:
    """
    uses the Yolo v3 algorithm to perform object detection
    Args:
        model_path is the path to where a Darknet Keras
        classes_path is the list of class names used for the Darknet
        class_t is a float representing the box score threshold
        nms_t is a float representing the IOU threshold
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = k.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
