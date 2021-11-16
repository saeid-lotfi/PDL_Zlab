# PDL_Zlab

Persian Digit Localizer implemented by Tensorflow and YOLOv3 model

## Installation
First, clone or download this GitHub repository.
Install requirements and download pretrained weights:
```
pip install -r ./requirements.txt

# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

# yolov3-tiny
wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights
```

## Data Generation
Start with generating data from raw digits and raw backgrounds to produce capcha like images with known bounding boxes and annotations. This module first generate examples in a voc-pascal format and then make its yolo format:
```
python 01-data-generation.py
```

## Setting Config
For changing train config we should edit yolov3.configs code in our way of interest.

## Train
`./yolov3/configs.py` file is already configured for digit localization.

Now, you can train it and then evaluate your model
```
python 02-train.py
tensorboard --logdir=log
```
Track training progress in Tensorboard and go to http://localhost:6006/:
<p align="center">
    <img width="100%" src="IMAGES/tensorboard.png" style="max-width:100%;"></a>
</p>

Test detection with `03-single-detect.py` script:
```
python 03-single-detect.py
```
