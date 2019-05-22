"""

Created on 2/05/18

"""

from typing import List
import cv2
import tensorflow as tf
import os
import numpy as np
from gv_tools.tracking.tracking_region import TrackingRegion
import time
import logging


class BodyDetector:
    def __init__(self, min_score=0.5, use_gpu=True, gpu_fraction: float = 1.0):
        self._gpu_fraction = gpu_fraction
        self._gpu_count = 1

        # Minimum score to consider as a detection.
        self.score_min = min_score
        # Tensorflow attributes.
        self._detection_graph = None
        self._session = None

        # Tensors.
        self._image_tensor = None
        self._detection_boxes = None
        self._detection_scores = None
        self._detection_classes = None
        self._num_detections = None

        # Disable or enable GPU.
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def process(self, image, upscale_width: int=0, upscale_height: int=0):
        """ Classify the input image and return the detections.
        Returns:
            tuple: Boxes (rect), scores (float), and classes (int).
        """

        # Resize before convert.
        cvt_images = image
        cvt_images = cv2.resize(cvt_images, (300, 300))   # Pre conversion seems a lot faster.
        cvt_images = cv2.cvtColor(cvt_images, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(cvt_images, axis=0)
        tmp_t = time.time()
        (boxes, scores, classes, num) = self._session.run(
            [self._detection_boxes, self._detection_scores, self._detection_classes, self._num_detections],
            feed_dict={self._image_tensor: image_np_expanded})
        logging.info(f'image passed through NN. {time.time() - tmp_t}')

        upscale_width = image.shape[1] if upscale_width == 0 else upscale_width
        upscale_height = image.shape[0] if upscale_height == 0 else upscale_height

        regions = []
        for i in range(len(boxes[0])):

            box = boxes[0][i]
            score = scores[0][i]
            class_id = classes[0][i].astype(np.int64)

            if score > self.score_min and class_id == 1:
                y_min, x_min, y_max, x_max = box
                face_region = TrackingRegion()

                # Get the absolute x and y limits for each box.
                face_region.set_rect(
                    left=int(x_min * upscale_width),
                    right=int(x_max * upscale_width),
                    top=int(y_min * upscale_height),
                    bottom=int(y_max * upscale_height)
                )

                face_region.confidence = float(score)
                face_region.data["class_id"] = "body"
                regions.append(face_region)

        return regions, [], []

    def load_model(self, path_to_model='/model_zoo/body.pb'):
        """ Load a TensorFlow frozen inference graph. This should only be used once."""

        self._detection_graph = tf.Graph()
        with self._detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self._gpu_fraction)
        config_proto = tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU': self._gpu_count})
        config_proto.gpu_options.allow_growth = True

        self._session = tf.Session(graph=self._detection_graph, config=config_proto)
        self._image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
        self._detection_boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')
        self._detection_scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
        self._detection_classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
        self._num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')

