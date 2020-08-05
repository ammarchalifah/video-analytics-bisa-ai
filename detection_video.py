#Plot object detection bounding boxes on a photo
#2020/07/2020
#by Ammar Chalifah

import matplotlib.pyplot as plt
import argparse
import cv2
import time
import os
import tensorflow as tf
import numpy as np

from functions import visualization_utils as vis_utils
from functions import label_map_util

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', default = 'ssd_mobilenet', help = 'Model name.')
parser.add_argument('-i', '--input_path', default = 'cars.mp4', help ='path of file')

args = parser.parse_args()

MODEL_NAME = args.model
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'models/'+ MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
# Number of classes to detect
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

start_time = time.time()

video = cv2.VideoCapture(args.input_path)

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Extract detection boxes
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Extract detection scores
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # Extract detection classes
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Extract number of detectionsd
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        while(video.isOpened()):
            ret, frame = video.read()
            frame_expanded = np.expand_dims(frame, axis=0)

            # Actual detection.
            (boxes_detections, scores_detections, classes_detections, num) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            image_np_with_detections = frame.copy()

            image_np_with_detections = vis_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                np.squeeze(boxes_detections),
                np.squeeze(classes_detections).astype(np.int32),
                np.squeeze(scores_detections),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)

            cv2.imshow('Object detector', image_np_with_detections)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        # Clean up
        video.release()
        cv2.destroyAllWindows()