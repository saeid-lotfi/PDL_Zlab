import os
import shutil
import random
import uuid
from PIL import Image
import numpy as np
from pascal_voc_writer import Writer
import configs
from configs import paths, files


# getting parameters
tfrecord_generator_script = 'tfrecord_generator.py'

def generate_example(digit_path, bg_path, output_path):
  '''
  generate one example from given raw digits and back grounds
  generated a new image and its annotation
  '''
  # annotation template
  example_name = str(uuid.uuid4()) # unique name for generated example
  writer = Writer(f"{output_path}/{example_name}.jpg", configs._image_size_[0], configs._image_size_[1])
  # make raw mask
  mask = Image.new(mode= 'L', size= configs._image_size_, color= 'white') # blank white image
  mask = np.array(mask) # changing to numpy
  # locating digits in mask
  n_digits = np.random.randint(1,6) # select number of digits in a single image
  raw_digits = os.listdir(digit_path) # list of available digits
  raw_backgrounds = os.listdir(bg_path) # list of available backgrounds
  if n_digits >= 1:
    for i in range(n_digits):
      # select random digit from raw digits
      random_digit = random.choice(raw_digits)
      digit_wh = np.random.randint(30, 120)
      # locate bounding box
      x_min = np.random.randint(int(configs._image_size_[0] * 0.05), int(configs._image_size_[0] * 0.7))
      y_min = np.random.randint(int(configs._image_size_[1] * 0.05), int(configs._image_size_[1] * 0.7))
      x_max = x_min + digit_wh
      y_max = y_min + digit_wh
      # open selected digit
      selected_digit = Image.open(f'{digit_path}/{random_digit}').convert('L') # converting to monocolor
      selected_digit = selected_digit.resize((digit_wh, digit_wh)) # resizing
      selected_digit = selected_digit.rotate(np.random.randint(-20, 20), fillcolor= 'white') # small transformation
      # editing mask
      mask[y_min:y_max, x_min:x_max] = np.array(selected_digit) # replacing mask with digit
      # adding annotation
      writer.addObject('digit', x_min, y_min, x_max, y_max)
  # save .xml annotation file
  writer.save(f"{output_path}/{example_name}.xml")
  # compose image
  mask = Image.fromarray(mask) # from numpy to Image
  fg_color = tuple(np.random.randint(256, size= 3)) # pick random color for digits
  fg_img = Image.new(mode= 'RGB', size= configs._image_size_, color= fg_color)
  random_back = random.choice(raw_backgrounds) # pick random background
  bg_img = Image.open(f"{bg_path}/{random_back}").resize(configs._image_size_) # opne and resize
  new_image = Image.composite(image1= bg_img, image2= fg_img, mask= mask)
  new_image.save(f"{output_path}/{example_name}.jpg", 'JPEG')
  

def data_generate(n_samples, digit_path, bg_path, output_path):
  for i in range(n_samples):
    generate_example(digit_path, bg_path, output_path)

# make train and test data
data_generate(n_samples= configs._number_of_train_samples_,
  digit_path= f"{paths['RAW_DIGITS']}/train",
  bg_path= f"{paths['RAW_BACKGROUNDS']}/train",
  output_path= paths['TRAIN_GENERATION_PATH'])
data_generate(n_samples= configs._number_of_test_samples_,
  digit_path= f"{paths['RAW_DIGITS']}/test",
  bg_path= f"{paths['RAW_BACKGROUNDS']}/test",
  output_path= paths['TEST_GENERATION_PATH'])


# make label map
with open(files['LABELMAP'], 'w') as f:
    for label in configs._labels_:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# creating tfrecords from images and labels
cmd = f"python {tfrecord_generator_script} -x {paths['TRAIN_GENERATION_PATH']} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}"
os.system(cmd)
cmd = f"python {tfrecord_generator_script} -x {paths['TEST_GENERATION_PATH']} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}"
os.system(cmd)
