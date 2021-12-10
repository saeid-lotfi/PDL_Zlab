from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
import os
from tqdm import tqdm
import lxml.etree

flags.DEFINE_string('classes', './data/digit.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_train_3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image_dir', './data/digit_data/JPEGImages', 'path to image directory')
flags.DEFINE_string('annot_dir', './data/digit_data/Annotations', 'path to annotation directory')
flags.DEFINE_string('image_list', './data/digit_data/ImageSets/val.txt', 'path to file of image names')
flags.DEFINE_string('output_detection', './evaluation/detections', 'path to detection output directory')
flags.DEFINE_string('output_groundtruth', './evaluation/groundtruths', 'path to groundtruth output directory')
flags.DEFINE_integer('num_classes', 10, 'number of classes in the model')

def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    os.makedirs(FLAGS.output_detection, exist_ok= True)
    os.makedirs(FLAGS.output_groundtruth, exist_ok= True)

    image_list = open(FLAGS.image_list).read().splitlines()
    for image_name in tqdm(image_list):
        # make groundtruth from xml annotation
        annot_xml = lxml.etree.fromstring(open(f"{FLAGS.annot_dir}/{image_name}.xml").read())
        annot_xml = parse_xml(annot_xml)['annotation']
        groundtruth_result = open(f"{FLAGS.output_groundtruth}/{image_name}.txt", "w")
        if annot_xml.get('object'):
            for element in annot_xml.get('object'):
                groundtruth_result.writelines(
                    [element['name'], ' ',
                    element['bndbox']['xmin'], ' ',
                    element['bndbox']['ymin'], ' ',
                    element['bndbox']['xmax'], ' ',
                    element['bndbox']['ymax'],
                    '\n'])
        groundtruth_result.close()

        # make detection with model
        img_raw = tf.image.decode_image(open(f"{FLAGS.image_dir}/{image_name}.jpg", 'rb').read(), channels=3)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)
        boxes, scores, classes, nums = yolo(img)
        boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]
        image_result = open(f"{FLAGS.output_detection}/{image_name}.txt", "w")
        wh = np.flip(img_raw.shape[0:2])
        for i in range(nums):
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            image_result.writelines([
                str(class_names[int(classes[i])]), ' ',
                str(np.array(scores[i])), ' ',
                str(x1y1[0]), ' ',
                str(x1y1[1]), ' ',
                str(x2y2[0]), ' ',
                str(x2y2[1]),
                '\n'])
        image_result.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
