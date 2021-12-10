from absl import app, flags, logging
from absl.flags import FLAGS
import os
import random
import uuid
from PIL import Image
import numpy as np
from pascal_voc_writer import Writer
from tqdm import tqdm


flags.DEFINE_string('data_dir', './data/digit_data',
                    'path to raw digit data')
flags.DEFINE_string('classes', './data/digit.names', 'classes file')
flags.DEFINE_integer('n_train', 1000, 'number of training samples')
flags.DEFINE_integer('n_val', 100, 'number of validation samples')
flags.DEFINE_integer('image_width', 640, 'image width')
flags.DEFINE_integer('image_height', 480, 'image height')
flags.DEFINE_integer('n_digit_in_image', 5, 'maximum number of digit in a single image')


def generate_image(digit_path, bg_path):
  """
  generate one example from given raw digits and back grounds
  generated a new image and its annotation
  """
  class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
  # annotation template
  example_name = str(uuid.uuid4()) # unique name for generated example
  writer = Writer(f"{FLAGS.data_dir}/JPEGImages/{example_name}.jpg", FLAGS.image_width, FLAGS.image_height)
  # make raw mask
  mask = Image.new(mode= 'L', size= [FLAGS.image_width, FLAGS.image_height], color= 'white') # blank white image
  mask = np.array(mask) # changing to numpy
  # locating digits in mask
  n_digits = np.random.randint(FLAGS.n_digit_in_image + 1) # select number of digits in a single image
  raw_digits = os.listdir(digit_path) # list of available digits
  raw_backgrounds = os.listdir(bg_path) # list of available backgrounds
  if n_digits >= 1:
    for i in range(n_digits):
      
      # select random digit from raw digits
      random_digit = random.choice(raw_digits) # random digit image
      label = class_names[int(random_digit[0])]
      selected_digit = Image.open(f'{digit_path}/{random_digit}').convert('L') # converting to monocolor
      digit_width, digit_height = (selected_digit.size[0], selected_digit.size[1]) # getting size
      random_scale = np.random.uniform(0.5, 1.5) # random choice for scaling
      digit_width = int(digit_width * random_scale)
      digit_height = int(digit_width * random_scale)
      selected_digit = selected_digit.resize((digit_width, digit_height)) # resizing
      selected_digit = selected_digit.rotate(np.random.randint(-10, 10), fillcolor= 'white') # small transformation
      # locate bounding box
      x_min = np.random.randint(low= 0, high= FLAGS.image_width - digit_width)
      y_min = np.random.randint(low= 0, high= FLAGS.image_height - digit_height)
      x_max = x_min + digit_width
      y_max = y_min + digit_height
      # editing mask
      mask[y_min:y_max, x_min:x_max] = np.minimum(np.array(selected_digit), mask[y_min:y_max, x_min:x_max]) # replacing mask with digit
      # adding annotation
      writer.addObject(label, x_min, y_min, x_max, y_max)
  # save .xml annotation file
  writer.save(f"{FLAGS.data_dir}/Annotations/{example_name}.xml")
  # compose image
  mask = Image.fromarray(mask) # from numpy to Image
  random_back = random.choice(raw_backgrounds) # pick random background
  bg_img = Image.open(f"{bg_path}/{random_back}").resize([FLAGS.image_width, FLAGS.image_height]) # opne and resize
  random_front = random.choice(list(set(raw_backgrounds) - set(random_back))) # pick random foreground
  fg_img = Image.open(f"{bg_path}/{random_front}").resize([FLAGS.image_width, FLAGS.image_height])
  random_color = tuple(np.random.randint(256, size= 3)) # pick random color for blending
  random_color = Image.new(mode= 'RGB', size= [FLAGS.image_width, FLAGS.image_height])
  fg_img = Image.blend(im1= fg_img, im2= random_color, alpha= 0.5)
  new_image = Image.composite(image1= bg_img, image2= fg_img, mask= mask)
  new_image.save(f"{FLAGS.data_dir}/JPEGImages/{example_name}.jpg", 'JPEG')
  
  return example_name
  

def main(_argv):

  # making project structure
  os.makedirs(f"{FLAGS.data_dir}/ImageSets", exist_ok= True)
  os.makedirs(f"{FLAGS.data_dir}/JPEGImages", exist_ok= True)
  os.makedirs(f"{FLAGS.data_dir}/Annotations", exist_ok= True)
  
  train_result = open(f"{FLAGS.data_dir}/ImageSets/train.txt", "a")
  val_result = open(f"{FLAGS.data_dir}/ImageSets/val.txt", "a")

  for i in tqdm(range(FLAGS.n_train)):
    image = generate_image(
      digit_path= f"{FLAGS.data_dir}/raw_digits/train",
      bg_path= f"{FLAGS.data_dir}/raw_backgrounds/train"
    )
    train_result.writelines([image, '\n'])
  logging.info(f"{FLAGS.n_train} samples generated for train")
  logging.info(f"train sample names stored in {FLAGS.data_dir}/ImageSets/train.txt")

  for i in tqdm(range(FLAGS.n_val)):
    image = generate_image(
      digit_path= f"{FLAGS.data_dir}/raw_digits/val",
      bg_path= f"{FLAGS.data_dir}/raw_backgrounds/val"
    )
    val_result.writelines([image, '\n'])
  logging.info(f"{FLAGS.n_val} samples generated for validation")
  logging.info(f"train sample names stored in {FLAGS.data_dir}/ImageSets/val.txt")
  
  train_result.close()
  val_result.close()



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

