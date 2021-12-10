# PDL_Zlab

Persian Digit Localizer implemented by Tensorflow and YOLOv3 model
*this repo is based on zzh8829/yolov3-tf2 https://github.com/zzh8829/yolov3-tf2.git

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
python tools/digit_data_generator.py \
--n_train 500 \
--n_val 100 \
--n_digit_in_image 5
```

## Convert to TFRecord
We need to convert raw images and .xml annotations to serialized tfrecord format:

```bash
python tools/digit_data_tfrecord.py
```

## Training
Start training with 'train.py' module:

``` bash
python train.py \
--batch_size 4 \
--dataset ./data/digits_train.tfrecord \
--val_dataset ./data/digits_val.tfrecord \
--epochs 50 --mode fit \
--transfer darknet
```

## Detection
Run detection with 'detect.py' module:

```bash
# custom trained yolov3
python detect.py \
--classes ./data/digit.names \
--num_classes 1 \
--weights ./checkpoints/yolov3_train_10.tf \
--image ./data/meme.jpg
```
