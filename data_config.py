import os

_number_of_train_samples_ = 500
_number_of_test_samples_ = 100
_image_size_ = (640, 640)
_labels_ = [{'name':'digit', 'id':1}]

paths = {
    'RAW_DIGITS': os.path.join('data', 'raw_digits'), # path to raw digits directory
    'RAW_BACKGROUNDS': os.path.join('data', 'raw_backgrounds'), # path to raw backgrounds directory
    'TRAIN_GENERATION_PATH': os.path.join('data', 'generated', 'train'), # path for generated train images and .xml files
    'TEST_GENERATION_PATH': os.path.join('data', 'generated', 'test'), # path for generated test images and .xml files
    'YOLO_FORMAT_BASE': os.path.join('data', 'yolo_format')
 }
