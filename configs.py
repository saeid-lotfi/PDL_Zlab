import os

_number_of_train_samples_ = 5
_number_of_test_samples_ = 3
_image_size_ = (640, 480)
_labels_ = [{'name':'digit', 'id':1}]

paths = {
    'RAW_DIGITS': os.path.join('data', 'raw_digits'), # path to raw digits directory
    'RAW_BACKGROUNDS': os.path.join('data', 'raw_backgrounds'), # path to raw backgrounds directory
    'TRAIN_GENERATION_PATH': os.path.join('Tensorflow', 'workspace', 'images', 'generated', 'train'), # path for generated train images and .xml files
    'TEST_GENERATION_PATH': os.path.join('Tensorflow', 'workspace', 'images', 'generated', 'test'), # path for generated test images and .xml files
    'LABELMAP': os.path.join('Tensorflow', 'workspace', 'annotations', 'label_map.pbtxt'), # label map file name and path
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'), # path to our workflow directory
    'APIMODEL_PATH': os.path.join('Tensorflow', 'models'), # path for Tensorflow Model Garden repo
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'), # path to tfrecord files and label maps
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'), # path for our trained models
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'pre-trained-models'), # path for pretrained models
 }
