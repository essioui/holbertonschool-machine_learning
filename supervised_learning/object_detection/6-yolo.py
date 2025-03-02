#!/usr/bin/env python3
"""
Module define Initialize Yolo
"""
from tensorflow import keras as k
import numpy as np
import cv2
import os


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

            threshold = box_score >= self.class_t

            filtered_boxes.append(box.reshape(-1, 4)[threshold.flatten()])
            box_classes.append(box_class.flatten()[threshold.flatten()])
            box_scores.append(box_score.flatten()[threshold.flatten()])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply Non-Maximum Suppression (NMS) for remove iteration boxes
        Args:
            filtered_boxes: array of shape (?,) containing filtered bounding
            box_classes: array containing class number for class filtered_boxe
            box_scores: array containing box scores for each box filtered_boxe
        Return:
            box_predictions
            predicted_box_classes
            predicted_box_scores
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            indx = np.where(box_classes == cls)

            cls_boxes = filtered_boxes[indx]
            cls_score = box_scores[indx]

            boxes_xywh = np.zeros_like(cls_boxes)
            boxes_xywh[:, 0] = cls_boxes[:, 0]  # x1
            boxes_xywh[:, 1] = cls_boxes[:, 1]  # y1
            boxes_xywh[:, 2] = cls_boxes[:, 2] - cls_boxes[:, 0]  # width
            boxes_xywh[:, 3] = cls_boxes[:, 3] - cls_boxes[:, 1]  # height

            indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(),
                                       cls_score.tolist(),
                                       self.class_t, self.nms_t)

            if len(indices) > 0:
                indices = indices.flatten()
                box_predictions.append(cls_boxes[indices])
                predicted_box_classes.append(np.full(len(indices), cls))
                predicted_box_scores.append(cls_score[indices])

        if len(box_predictions) == 0:
            return np.array([]), np.array([]), np.array([])

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from the given folder path.
        Args:
        folder_path (str): Path to the folder containing images.
        Returns:
        tuple: (images, image_paths)
            - images: list of images as numpy.ndarrays
            - image_paths: list of image file paths
        """
        image_paths = []
        images = []

        # Get all image files from the directory
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            # Check if the file is an image
            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(file_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images before input to model
        Args:
            images: a list of images as numpy.ndarray
        Return
            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                ni: the number of images
                input_h: the input height for the Darknet model
                input_w: the input width for the Darknet model
                3: number of color channels
            image_shapes: a numpy.ndarray of shape (ni, 2)
        """
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        lpimages = []
        limage_shapes = []

        for img in images:

            img_shape = img.shape[0], img.shape[1]
            limage_shapes.append(img_shape)

            dimension = (input_w, input_h)
            resized = cv2.resize(img, dimension,
                                 interpolation=cv2.INTER_CUBIC)

            pimage = resized / 255
            lpimages.append(pimage)

        pimages = np.array(lpimages)
        image_shapes = np.array(limage_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Show image with bounding bow parameters
        Args:
            image: containing an unprocessed image
            boxes: containing the boundary boxes for the imag
            box_classes: containing the class indices for each box
            box_scores: containing the box scores for each box
            file_name: the file path where the original image is stored
        Return:
            images with bounding boxes
            save the image in folder detections
        """
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            class_id = box_classes[i]
            score = box_scores[i]

            label = f"{self.class_names[class_id]}: {score: 2f}"

            # design bounding box with bleu color
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # position of text
            text_position = (x1, max(y1 - 5, 10))

            # design the text with red color
            cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # show image one time
        cv2.imshow(file_name, image)

        # wait key input from user
        key = cv2.waitKey(0)

        # create new folder if isnt find
        if key == ord('s'):
            os.makedirs("detections", exist_ok=True)

            save_path = os.path.join("detections", file_name)

            cv2.imwrite(save_path, image)

            print(f"the image is saved: {save_path}")

        # close all window
        cv2.destroyAllWindows()
