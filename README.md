# PDL_Zlab

Persian Digit Localizer implemented by Tensorflow and YOLOv3 model
*this repo is based on:


zzh8829/yolov3-tf2 https://github.com/zzh8829/yolov3-tf2.git
rafaelpadilla/Object-Detection-Metrics https://github.com/rafaelpadilla/Object-Detection-Metrics

## Installation
First, clone or download this GitHub repository and install requirements:

```bash
pip install -r ./requirements.txt
```

## Download and convert pre-trained Darknet weights

```bash
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights

python tools/convert_darknet.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights

python tools/convert_darknet.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny
```

## Data Generation
Start with generating data from raw digits and raw backgrounds to produce capcha like images with known bounding boxes and annotations. This module generate examples in voc-structured repository.

```bash
# train
python tools/digit_data_generator.py \
--digit_dir ./data/digit_data/raw_digits/train \
--bg_dir ./data/digit_data/raw_image/Dark_soft \
--fg_dir ./data/digit_data/raw_image/Light_soft \
--dataset_split train \
--n_samples 500
python tools/digit_data_generator.py \
--digit_dir ./data/digit_data/raw_digits/train \
--bg_dir ./data/digit_data/raw_image/Light_soft \
--fg_dir ./data/digit_data/raw_image/Dark_soft \
--dataset_split train \
--n_samples 500
# val
python tools/digit_data_generator.py \
--digit_dir ./data/digit_data/raw_digits/val \
--bg_dir ./data/digit_data/raw_image/Dark_soft \
--fg_dir ./data/digit_data/raw_image/Light_soft \
--dataset_split val \
--n_samples 100
python tools/digit_data_generator.py \
--digit_dir ./data/digit_data/raw_digits/val \
--bg_dir ./data/digit_data/raw_image/Light_soft \
--fg_dir ./data/digit_data/raw_image/Dark_soft \
--dataset_split val \
--n_samples 100
```

## Convert to TFRecord
We need to convert raw images and .xml annotations to serialized tfrecord format:

```bash
# train
python tools/digit_data_tfrecord.py \
--output_file ./data/digits_train.tfrecord \
--split train
# val
python tools/digit_data_tfrecord.py \
--output_file ./data/digits_val.tfrecord \
--split val
```

## Training
Start training with 'train.py' module:

``` bash
# training custom yolov3
python train.py \
--classes ./data/digit.names \
--dataset ./data/digits_train.tfrecord \
--val_dataset ./data/digits_val.tfrecord \
--epochs 50 \
--mode fit \
--transfer darknet \
--num_classes 10 \
--weights_num_classes 80 \
--batch_size 4 \
```

## Detection
Run detection with 'detection.py' module:

```bash
# custom trained yolov3
python detection.py \
--classes ./data/digit.names \
--num_classes 10 \
--weights ./checkpoints/yolov3_train_10.tf
```

## Evaluation
Run evaluation with 'evaluatior.py' module:

```bash
# pascal voc metrics
python evaluator.py \
-imgsize 640,480
```