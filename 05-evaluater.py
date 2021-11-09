import os
import configs
from configs import paths

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CHECKPOINT_PATH = os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME) # path for our specific model
PIPELINE_FILE = os.path.join(CHECKPOINT_PATH, 'pipeline.config')

# making evaluation command for TFOD
EVALUATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
command = f"python {EVALUATION_SCRIPT} --model_dir={CHECKPOINT_PATH} --pipeline_config_path={PIPELINE_FILE} --checkpoint_dir={CHECKPOINT_PATH}"
os.system(command)
