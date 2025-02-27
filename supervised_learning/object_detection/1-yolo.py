#!/usr/bin/env python3
"""
Module define Initialize Yolo
"""
from tensorflow import keras as k
import numpy as np


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

    def sigmoid(self, x):
        """
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        this function has return from DarkNet:
        boxes:
            a list of shape (grid_height, grid_width, anchor_boxes, 4)
        box_confidences:
            a list of shape (grid_height, grid_width, anchor_boxes, 1)
        box_class_probs:
            a list of shape (grid_height, grid_width, anchor_boxes, classes)
        """
        image_height, image_width = image_size
        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            tx_ty = self.sigmoid(output[..., :2])
            th_tw = np.exp(output[..., 2:4])

            cx = np.tile(np.arange(grid_width).reshape(1, -1, 1),
                         (grid_height, 1, anchor_boxes))
            cy = np.tile(np.arange(grid_height).reshape(-1, 1, 1),
                         (1, grid_width, anchor_boxes))

            bx = (tx_ty[..., 0] + cx) / grid_width
            by = (tx_ty[..., 1] + cy) / grid_height

            anchor_w = self.anchors[i][:, 0].reshape(1, 1, anchor_boxes)
            anchor_h = self.anchors[i][:, 1].reshape(1, 1, anchor_boxes)

            bw = (th_tw[..., 0] * anchor_w) / image_width
            bh = (th_tw[..., 1] * anchor_h) / image_height

            x1 = (bx - (bw / 2)) * image_width
            y1 = (by - (bh / 2)) * image_height
            x2 = (bx - (bw / 2)) * image_width
            y2 = (by - (bh / 2)) * image_height

            processed_boxes = np.stack([x1, y1, x2, y2], axis=-1)

            box_confidence = self.sigmoid(output[..., 4:5])

            box_class_probilities = self.sigmoid(output[..., 5:])

            boxes.append(processed_boxes)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_probilities)

        return boxes, box_confidences, box_class_probs
