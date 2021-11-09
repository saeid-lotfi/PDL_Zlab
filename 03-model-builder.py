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
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
CHECKPOINT_PATH = os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME) # path for our specific model
PIPELINE_FILE = os.path.join(CHECKPOINT_PATH, 'pipeline.config')
LABEL_FILE = os.path.join(paths['ANNOTATION_PATH'], 'label_map.pbtxt')

# get pretrained model
pretrained_model = wget.download(PRETRAINED_MODEL_URL) # download pretrained model
shutil.move(pretrained_model, paths['PRETRAINED_MODEL_PATH']) # move to dedicated directory inside workspace
# unzip files
tar = tarfile.open(f"{paths['PRETRAINED_MODEL_PATH']}/{pretrained_model}", "r:gz")
tar.extractall(paths['PRETRAINED_MODEL_PATH'])
tar.close()

# creating a template from a pretrained model in our custom model directory
os.makedirs(CHECKPOINT_PATH, exist_ok= True) # make our directory
shutil.copy(
    src= os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'),
    dst= os.path.join(CHECKPOINT_PATH, 'pipeline.config'))


# getting config and change to protobuf format
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig() # make blank template
with tf.io.gfile.GFile(PIPELINE_FILE, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# change pipeline config to our desired setting
pipeline_config.model.ssd.num_classes = len(configs._labels_)
pipeline_config.train_config.batch_size = 2
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= LABEL_FILE
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = LABEL_FILE
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

# write back new config file
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(PIPELINE_FILE, "wb") as f:
    f.write(config_text)
