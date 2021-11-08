import os

_number_of_train_samples_ = 5
_number_of_test_samples_ = 3
_image_size_ = (640, 480)
_labels_ = [{'name':'digit', 'id':1}]

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'RAW_DIGITS': os.path.join('data','raw_digits'),
    'RAW_BACKGROUNDS': os.path.join('data','raw_backgrounds'),
    'WORKSPACE_PATH': os.path.join('Tensorflow','workspace'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow','workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow','workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow','workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow','workspace','pre-trained-models'),
    'PROTOC_PATH': os.path.join('Tensorflow','protoc'),
    'TRAIN_GENERATION_PATH': os.path.join('Tensorflow','workspace','images','generated','train'),
    'TEST_GENERATION_PATH': os.path.join('Tensorflow','workspace','images','generated','test')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], 'label_map.pbtxt')
}
