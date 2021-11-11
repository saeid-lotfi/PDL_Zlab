import os
from configs import paths

# making project structure
for path in paths.values():
    os.makedirs(path, exist_ok= True)

# cloning and installing TFOD API
os.system(f"git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}") # clone github
os.system('apt-get install protobuf-compiler') # install protobuf compiler
os.system('cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . ')

# installing requirements
os.system('pip install -r requirements.txt')
os.system('pip install -U --pre tensorflow=="2.2.0"')

# Verify Installation
VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
os.system(f"python {VERIFICATION_SCRIPT}")