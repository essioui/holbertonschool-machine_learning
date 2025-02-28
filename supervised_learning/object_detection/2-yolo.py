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
        Processes the outputs from the YOLO model.

        Args:
            outputs (List[np.ndarray]): List of arrays model outputs.
            image_size (Tuple[int, int]): (image_height, image_width).

        Returns:
            [List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                - boxes: List of bounding boxes
                    (grid_height, grid_width, anchor_boxes, 4).
                - box_confidences: List of box confidence scores
                    (grid_height, grid_width, anchor_boxes, 1).
                - box_class_probs: List of class probability scores
                    (grid_height, grid_width, anchor_boxes, classes).
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        img_h, img_w = image_size

        for output in outputs:
            boxes.append(output[..., 0:4])
            box_confidences.append(self.sigmoid(output[..., 4, np.newaxis]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))
        for i, box in enumerate(boxes):
            gr_h, gr_w, anchors_boxes, _ = box.shape
            cx = np.indices((gr_h, gr_w, anchors_boxes))[1]
            cy = np.indices((gr_h, gr_w, anchors_boxes))[0]
            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]
            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]
            bx = (self.sigmoid(t_x) + cx) / gr_w
            by = (self.sigmoid(t_y) + cy) / gr_h
            bw = (np.exp(t_w) * p_w) / self.model.input.shape[1]
            bh = (np.exp(t_h) * p_h) / self.model.input.shape[2]
            tl_x = bx - bw / 2
            tl_y = by - bh / 2
            br_x = bx + bw / 2
            br_y = by + bh / 2
            box[..., 0] = tl_x * img_w
            box[..., 1] = tl_y * img_h
            box[..., 2] = br_x * img_w
            box[..., 3] = br_y * img_h
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes with hight confiences
        Args:
            boxes:
                numpy.ndarrays of shape(grid_height,grid_width,anchor_boxes,4)
            box_confidences:
                numpy.ndarrays of shape(grid_height,grid_width,anchor_boxes,1)
            box_class_probs:
                numpy.ndarrays (grid_height, grid_width, anchor_boxes, classes)
            Return:
                tuple: (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes, box_classes, box_scores = [], [], []

        for box, conf, prob in zip(boxes, box_confidences, box_class_probs):
            scores = prob * conf

            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)

            filtered_boxes.append(box.reshape(-1, 4))
            box_classes.append(box_class.flatten())
            box_scores.append(box_score.flatten())

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
