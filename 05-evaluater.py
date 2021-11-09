import os
import shutil
import wget
import tarfile
import object_detection
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import configs
from configs import paths

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CHECKPOINT_PATH = os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME) # path for our specific model
PIPELINE_FILE = os.path.join(CHECKPOINT_PATH, 'pipeline.config')
LABEL_FILE = os.path.join(paths['ANNOTATION_PATH'], 'label_map.pbtxt')
TRAIN_NUM = configs._train_number_step_

# making evaluation command for TFOD
EVALUATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
command = f"python {EVALUATION_SCRIPT} --model_dir={CHECKPOINT_PATH} --pipeline_config_path={PIPELINE_FILE} --checkpoint_dir={CHECKPOINT_PATH}"
os.system(command)
