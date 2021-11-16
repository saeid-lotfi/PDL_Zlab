import os
import glob
import shutil
import random
import uuid
from PIL import Image
import numpy as np
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET
from data_config import paths
import data_config


# making project structure
for path in paths.values():
    os.makedirs(path, exist_ok= True)

def generate_example(digit_path, bg_path, output_path):
  '''
  generate one example from given raw digits and back grounds
  generated a new image and its annotation
  '''
  # annotation template
  example_name = str(uuid.uuid4()) # unique name for generated example
  writer = Writer(f"{output_path}/{example_name}.jpg", data_config._image_size_[0], data_config._image_size_[1])
  # make raw mask
  mask = Image.new(mode= 'L', size= data_config._image_size_, color= 'white') # blank white image
  mask = np.array(mask) # changing to numpy
  # locating digits in mask
  n_digits = np.random.randint(4) # select number of digits in a single image
  raw_digits = os.listdir(digit_path) # list of available digits
  raw_backgrounds = os.listdir(bg_path) # list of available backgrounds
  if n_digits >= 1:
    for i in range(n_digits):
      
      # select random digit from raw digits
      random_digit = random.choice(raw_digits) # random digit image
      selected_digit = Image.open(f'{digit_path}/{random_digit}').convert('L') # converting to monocolor
      digit_width, digit_height = (selected_digit.size[0], selected_digit.size[1]) # getting size
      random_scale = np.random.uniform(0.5, 1.5) # random choice for scaling
      digit_width = int(digit_width * random_scale)
      digit_height = int(digit_width * random_scale)
      selected_digit = selected_digit.resize((digit_width, digit_height)) # resizing
      selected_digit = selected_digit.rotate(np.random.randint(-10, 10), fillcolor= 'white') # small transformation
      # locate bounding box
      x_min = np.random.randint(int(data_config._image_size_[0] * 0.05), int(data_config._image_size_[0] * 0.6))
      y_min = np.random.randint(int(data_config._image_size_[1] * 0.05), int(data_config._image_size_[1] * 0.6))
      x_max = x_min + digit_width
      y_max = y_min + digit_height
      # editing mask
      mask[y_min:y_max, x_min:x_max] = np.array(selected_digit) # replacing mask with digit
      # adding annotation
      writer.addObject('digit', x_min, y_min, x_max, y_max)
  # save .xml annotation file
  writer.save(f"{output_path}/{example_name}.xml")
  # compose image
  mask = Image.fromarray(mask) # from numpy to Image
  fg_color = tuple(np.random.randint(256, size= 3)) # pick random color for digits
  fg_img = Image.new(mode= 'RGB', size= data_config._image_size_, color= fg_color)
  random_back = random.choice(raw_backgrounds) # pick random background
  bg_img = Image.open(f"{bg_path}/{random_back}").resize(data_config._image_size_) # opne and resize
  new_image = Image.composite(image1= bg_img, image2= fg_img, mask= mask)
  new_image.save(f"{output_path}/{example_name}.jpg", 'JPEG')
  

def data_generate(n_samples, digit_path, bg_path, output_path):
  for i in range(n_samples):
    generate_example(digit_path, bg_path, output_path)

def ParseXML(img_folder, file):
    for xml_file in glob.glob(img_folder+'/*.xml'):
        tree=ET.parse(open(xml_file))
        root = tree.getroot()
        image_name = root.find('filename').text
        img_path = img_folder+'/'+image_name
        for i, obj in enumerate(root.iter('object')):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in Dataset_names:
                Dataset_names.append(cls)
            cls_id = Dataset_names.index(cls)
            xmlbox = obj.find('bndbox')
            OBJECT = (str(int(float(xmlbox.find('xmin').text)))+','
                      +str(int(float(xmlbox.find('ymin').text)))+','
                      +str(int(float(xmlbox.find('xmax').text)))+','
                      +str(int(float(xmlbox.find('ymax').text)))+','
                      +str(cls_id))
            img_path += ' '+OBJECT
        # print(img_path)
        file.write(img_path+'\n')

def run_XML_to_YOLOv3():
    for i, folder in enumerate(['train','test']):
        with open([Dataset_train,Dataset_test][i], "w") as file:
            # print(os.getcwd()+data_dir+folder)
            img_path = os.path.join(os.getcwd()+data_dir+folder)
            if is_subfolder:
                for directory in os.listdir(img_path):
                    xml_path = os.path.join(img_path, directory)
                    ParseXML(xml_path, file)
            else:
                ParseXML(img_path, file)

    print("Dataset_names:", Dataset_names)
    with open(Dataset_names_path, "w") as file:
        for name in Dataset_names:
            file.write(str(name)+'\n')

# make train and test data
data_generate(n_samples= data_config._number_of_train_samples_,
  digit_path= f"{paths['RAW_DIGITS']}/train",
  bg_path= f"{paths['RAW_BACKGROUNDS']}/train",
  output_path= paths['TRAIN_GENERATION_PATH'])
data_generate(n_samples= data_config._number_of_test_samples_,
  digit_path= f"{paths['RAW_DIGITS']}/test",
  bg_path= f"{paths['RAW_BACKGROUNDS']}/test",
  output_path= paths['TEST_GENERATION_PATH'])

# export to yolo format
os.makedirs(paths['YOLO_FORMAT_BASE'], exist_ok= True)
data_dir = '/data/generated/'
Dataset_names_path = os.path.join(paths['YOLO_FORMAT_BASE'], 'digit_names.txt')
Dataset_train = os.path.join(paths['YOLO_FORMAT_BASE'], 'digit_train.txt')
Dataset_test = os.path.join(paths['YOLO_FORMAT_BASE'], 'digit_test.txt')
is_subfolder = False

Dataset_names = []

run_XML_to_YOLOv3()
