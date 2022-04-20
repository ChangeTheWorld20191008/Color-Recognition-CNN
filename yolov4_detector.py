import cv2
import colorsys
import random
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import ConfigProto
from tensorflow.python.saved_model import tag_constants


class ObjectDetector:
    def __init__(self, model_path='./model', label_file='./model/label.names',
                 num_classes=2, score_threshold=0.3, image_sz=(416, 416, 3)):
        self._model_path = model_path
        self._label_file = label_file
        self._num_classes = num_classes
        self._score_threshold = score_threshold
        self._image_sz = image_sz[0:2]

        self._config = ConfigProto()
        self._config.gpu_options.allow_growth = True

        self._graph = tf.Graph()

        with self._graph.as_default():
            self._sess = tf.Session(config=self._config)

            tf.saved_model.load(
                self._sess, [tag_constants.SERVING], self._model_path)

            self._image_tensor = self._sess.graph.get_tensor_by_name(
                'serving_default_input_1:0')
            self._output_tensor = self._sess.graph.get_tensor_by_name(
                'StatefulPartitionedCall:0')

            self._boxes = tf.placeholder(
                tf.float32, shape=(None, None, None, 4))
            self._scores = tf.placeholder(
                tf.float32, shape=(None, None, self._num_classes))

            self._boxes_predi, self._scores_predi, self._classes_predi,\
                self._valid_detections_predi = \
                tf.image.combined_non_max_suppression(
                    boxes=self._boxes, scores=self._scores,
                    max_output_size_per_class=50, max_total_size=50,
                    iou_threshold=0.45, score_threshold=self._score_threshold)

            self._label_map = self._load_labelmap(self._label_file)

    def _load_labelmap(self, label_file):
        category_index = {}
        index = 1

        for line in open(label_file):
            category_index[index] = line.rstrip("\n")
            index += 1

        return category_index

    def feed(self, image):
        image_h, image_w, _ = image.shape
        ori_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(ori_image, self._image_sz)
        det_image = image_data / 255.
        image_np_expanded = np.expand_dims(det_image, axis=0)
        image_np_expanded = np.asarray(image_np_expanded).astype(np.float32)

        pred_bbox = self._sess.run(
            self._output_tensor,
            feed_dict={self._image_tensor: image_np_expanded})

        boxes_pred, scores_pred, classes_pred, valid_detections_pred = \
            self._sess.run(
                [self._boxes_predi, self._scores_predi, self._classes_predi,
                 self._valid_detections_predi],
                feed_dict={
                    self._boxes: np.reshape(
                        pred_bbox[:, :, 0:4],
                        (pred_bbox[:, :, 0:4].shape[0], -1, 1, 4)),
                    self._scores: pred_bbox[:, :, 4:]})

        boxes = boxes_pred[0][:valid_detections_pred[0]]
        scores = scores_pred[0][:valid_detections_pred[0]]
        classes = classes_pred[0][:valid_detections_pred[0]] + 1
        labels = [self._label_map[classes_id] for classes_id in classes]

        objects = []
        for i in range(len(boxes)):
            detection_info = dict()
            detection_info['class_id'] = classes[i]
            detection_info['confidence'] = scores[i]
            detection_info['label'] = labels[i]
            detection_info['ymin'] = int(boxes[i][0] * image_h)
            detection_info['xmin'] = int(boxes[i][1] * image_w)
            detection_info['ymax'] = int(boxes[i][2] * image_h)
            detection_info['xmax'] = int(boxes[i][3] * image_w)
            objects.append(detection_info)

        return objects

    def close(self):
        self._sess.close()
        self._sess = None


def draw_bbox(image, boxes, scores, labels):
    image_h, image_w, _ = image.shape
    fontScale = 0.5
    bbox_color = (0, 255, 0)

    for i in range(len(boxes)):
        coor = boxes[i]
        score = scores[i]
        label = labels[i]

        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        bbox_mess = \
            '%s: %.2f' % (label, score)
        t_size = cv2.getTextSize(
            bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(
            image, c1, c3,
            bbox_color, -1)  # filled

        cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0),
                    bbox_thick // 2, lineType=cv2.LINE_AA)
    return image


if __name__ == '__main__':
    model_path = '/home/zhuhao/myModel/person_and_car/tiny'
    label_file = '/home/zhuhao/myModel/person_and_car/tiny/label.names'

    image_path = '/home/zhuhao/dataset/person_and_car/office'
    output_dir = '/home/zhuhao/dataset/tmp/results/tiny'

    object_detector = ObjectDetector(model_path, label_file)

    image_list = os.listdir(image_path)
    for image_file in image_list:
        image = cv2.imread(f"{image_path}/{image_file}")
        objects = object_detector.feed(image)

        boxes = []
        scores = []
        labels = []
        for object in objects:
            print(f"[INFO]: object is {object}")
            box = [0, 0, 0, 0]
            box[0] = object['ymin']
            box[1] = object['xmin']
            box[2] = object['ymax']
            box[3] = object['xmax']
            boxes.append(box)
            scores.append(object['confidence'])
            labels.append(object['label'])
        out_image = draw_bbox(image, boxes, scores, labels)

        out_image = Image.fromarray(out_image.astype(np.uint8))
        out_image = np.array(out_image)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(os.path.join(output_dir, image_file), out_image)

    object_detector.close()
