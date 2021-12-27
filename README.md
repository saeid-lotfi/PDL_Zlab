# PDL_Zlab

Persian Digit Localizer using YOLOv3 model implemented by Tensorflow

*this repo is based on:
- zzh8829/yolov3-tf2 https://github.com/zzh8829/yolov3-tf2.git
- rafaelpadilla/Object-Detection-Metrics https://github.com/rafaelpadilla/Object-Detection-Metrics

## Installation
First, clone or download this GitHub repository and install requirements:

```bash
pip install -r ./requirements.txt
```

## Download and convert pre-trained Darknet weights
For transfer learning we need pre-trained weights. We get these weights which are in another framework "Darknet" and then convert it for using inside tensorflow framework as a pre-trained feature extractor. There are two types of YOLOv3 model:
- default one which is larger
- tiny one which is much smaller and therefore less accurate

```bash
# downloading yolov3 darknet weights
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
# converting to tensorflow model weights
python tools/convert_darknet.py \
--weights ./data/yolov3.weights \
--output ./checkpoints/yolov3.tf

# downloading yolov3-tiny darknet weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights
# converting to tensorflow model weights
python tools/convert_darknet.py \
--weights ./data/yolov3-tiny.weights \
--output ./checkpoints/yolov3-tiny.tf --tiny
```

## Data Generation
Start with generating data from raw digits and raw backgrounds to generate synthetic capcha like images with known bounding boxes and annotations. This module generates examples in VOC-structured repository. Three parameters should be given to this code:
- Digit directory, consisting of many single digit images all in JPEG format
- Background directory, images in this directory are used as a base image in generation
- Foreground directory, images in this directory are used for digit fillings pattern blended with random color

After running, three extra directories are created inside digit data directory: one for all JPEG images, one for all annotations in XML format, and another for set records in text format keeping example names.

```bash
# training data generation
# using dark background
python tools/digit_data_generator.py \
--digit_dir ./data/digit_data/raw_digits/train \
--bg_dir ./data/digit_data/raw_image/Dark_soft \
--fg_dir ./data/digit_data/raw_image/Light_soft \
--dataset_split train \
--n_samples 500
# using light backgrounds
python tools/digit_data_generator.py \
--digit_dir ./data/digit_data/raw_digits/train \
--bg_dir ./data/digit_data/raw_image/Light_soft \
--fg_dir ./data/digit_data/raw_image/Dark_soft \
--dataset_split train \
--n_samples 500

# validation data generation
# using dark background
python tools/digit_data_generator.py \
--digit_dir ./data/digit_data/raw_digits/val \
--bg_dir ./data/digit_data/raw_image/Dark_soft \
--fg_dir ./data/digit_data/raw_image/Light_soft \
--dataset_split val \
--n_samples 100
# using light backgrounds
python tools/digit_data_generator.py \
--digit_dir ./data/digit_data/raw_digits/val \
--bg_dir ./data/digit_data/raw_image/Light_soft \
--fg_dir ./data/digit_data/raw_image/Dark_soft \
--dataset_split val \
--n_samples 100
```

## Convert to TFRecord
Data needs to convert from VOC-structured data to serialized tfrecord format before training.

```bash
# training data
python tools/digit_data_tfrecord.py \
--output_file ./data/digits_train.tfrecord \
--split train
# validation data
python tools/digit_data_tfrecord.py \
--output_file ./data/digits_val.tfrecord \
--split val
```

## Training
Start training with 'train.py' module:

``` bash
# training custom yolov3 model based on darknet feature extractor
python train.py \
--classes ./data/digit.names \
--dataset ./data/digits_train.tfrecord \
--val_dataset ./data/digits_val.tfrecord \
--epochs 50 \
--mode fit \
--transfer darknet \
--num_classes 10 \
--weights_num_classes 80 \
--batch_size 4
```

## Detection and Evaluation
When there is a trained model, run this module to perform a detection on the test dataset. Dataset should contain
- a directory for JPEG images
- a directory for XML files
- and a text file keeping image names.

After running this code, evaluation direcory is created in main directory and consist of detection result and groundtruth result. For every image, there is a text file in detection directory that shows the trained model detection on that image, and there is another text file in groundtruth directory which is the real label. Next module uses these two directories to evaluate the model and show different metrics.

```bash
# detection
python detection.py \
--classes ./data/digit.names \
--num_classes 10 \
--weights ./checkpoints/yolov3_train_?.tf
# evaluation
python evaluator.py
```