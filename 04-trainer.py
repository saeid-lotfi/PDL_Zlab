import os
import configs
from configs import paths

CUSTOM_MODEL_NAME = 'my_det'
CHECKPOINT_PATH = os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME) # path for our specific model
PIPELINE_FILE = os.path.join(CHECKPOINT_PATH, 'pipeline.config')
LABEL_FILE = os.path.join(paths['ANNOTATION_PATH'], 'label_map.pbtxt')
TRAIN_NUM = configs._train_number_step_

# making train command for TFOD
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
command = f"python {TRAINING_SCRIPT} --model_dir={CHECKPOINT_PATH} --pipeline_config_path={PIPELINE_FILE} --num_train_steps={TRAIN_NUM}"
os.system(command)
