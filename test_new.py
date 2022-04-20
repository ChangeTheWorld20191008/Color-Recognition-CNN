import os

import numpy as np
import cv2
from keras.models import load_model
from keras.optimizers import SGD
from PIL import Image

from yolov4_detector import ObjectDetector
from yolov4_detector import draw_bbox


labels_file = 'labels_person_and_car.txt'
model_path = '/home/zhuhao/myModel/person_and_car/yolov4_tf_person_car_v3'
image_path = '/home/zhuhao/dataset/car/color/tmp'
output_dir = '/home/zhuhao/dataset/car/color/results'

object_detector = ObjectDetector(model_path, labels_file)

labels = ['black', 'blue', 'brown', 'green',
          'pink', 'red', 'silver', 'white', 'yellow']

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model = load_model('color_model.h5')
model.compile(
    optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

image_list = os.listdir(image_path)
image_number = len(image_list)
for i, image_file in enumerate(image_list, 1):
    print(f"[INFO]: {i} - {image_file} is processing, left {image_number-i}")
    image = cv2.imread(f"{image_path}/{image_file}")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    objects = object_detector.feed(image)

    boxes = []
    scores = []
    titles = []
    for object in objects:
        if object['label'] == 'car' and object['confidence'] >= 0.5:
            ymin = object['ymin']
            xmin = object['xmin']
            ymax = object['ymax']
            xmax = object['xmax']

            box = [ymin, xmin, ymax, xmax]
            boxes.append(box)
            scores.append(object['confidence'])

            car_crop_image = rgb_image[ymin:ymax, xmin:xmax]

            img = cv2.resize(car_crop_image, (224, 224))

            x = np.expand_dims(img, axis=0)

            classes = model.predict(x, batch_size=1)
            print(f"[TMP]: classes are {classes}")
            color = labels[np.argmax(classes)]

            titles.append(f"{object['label']}:{color}")

    out_image = draw_bbox(image, boxes, scores, titles)

    out_image = Image.fromarray(out_image.astype(np.uint8))
    out_image = np.array(out_image)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, image_file), out_image)

object_detector.close()
