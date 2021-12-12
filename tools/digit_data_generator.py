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
                    'path to data directory')
flags.DEFINE_string('digit_dir', '', 'path to raw digit directory')
flags.DEFINE_string('bg_dir', '', 'path to raw backgrounds directory')
flags.DEFINE_string('fg_dir', '', 'path to raw foregrounds directory')
flags.DEFINE_string('classes', './data/digit.names', 'classes file')
flags.DEFINE_string('dataset_split', 'train', 'dataset partition name')
flags.DEFINE_integer('n_samples', 500, 'number of samples')
flags.DEFINE_integer('image_width', 640, 'image width')
flags.DEFINE_integer('image_height', 480, 'image height')
flags.DEFINE_integer('n_digit_in_image', 4, 'maximum number of digit in a single image')

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb : (x_min, y_min, x_max, y_max)

    """
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

def bb_intersection_check(bounding_box, bounding_box_list, threshold):
    """
    Check if bounding_box has no conflict with other bounding boxes

    Parameters
    ----------
    bounding_box : (x_min, y_min, x_max, y_max)
    bounding_box_list : [bb1, bb2, ...]
    threshold : threshold for deciding conflict
    ----------
    return true if a bounding box can be added to list of other bounding boxes
    """
    if len(bounding_box_list) == 0:
        return True
    else:
        for bb in bounding_box_list:
            if get_iou(bb, bounding_box) >= threshold:
                return False
    return True

def bb_generator(image_size, digit_size, bb_list):
    """
    Generate a new bounding box

    Parameters
    ----------
    image_size : (width, height)
    digit_size : (width, height)
    bb_list : [bb1, bb2, ...]
    ----------
    return bb : (x_min, y_min, x_max, y_max)

    """
    while True:
        x_min = np.random.randint(low= 0, high= image_size[0] - digit_size[0])
        y_min = np.random.randint(low= 0, high= image_size[1] - digit_size[1])
        x_max = x_min + digit_size[0]
        y_max = y_min + digit_size[1]
        bb = (x_min, y_min, x_max, y_max)
        if bb_intersection_check(bb, bb_list, threshold= 0.15):
            break
    return bb

def generate_image(digit_path, bg_path, fg_path):
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
    raw_foregrounds = os.listdir(fg_path) # list of available foregrounds
    bounding_box_list = [] # keep a list of image bounding boxes
    if n_digits >= 1:
        for i in range(n_digits):
        
            # select random digit from raw digits
            random_digit = random.choice(raw_digits) # random digit image
            digit_label = class_names[int(random_digit[0])] # digit label
            selected_digit = Image.open(f'{digit_path}/{random_digit}').convert('L') # converting to monocolor
            digit_width, digit_height = (selected_digit.size[0], selected_digit.size[1]) # getting size
            random_scale = np.random.uniform(0.5, 1.5) # random choice for scaling
            digit_width = int(digit_width * random_scale)
            digit_height = int(digit_width * random_scale)
            selected_digit = selected_digit.resize((digit_width, digit_height)) # resizing
            selected_digit = selected_digit.rotate(np.random.randint(-10, 10), fillcolor= 'white') # small transformation
            # generate a possible bounding box
            bb = bb_generator(
                image_size= (FLAGS.image_width, FLAGS.image_height),
                digit_size= (digit_width, digit_height), 
                bb_list= bounding_box_list)
            # add the bounding box to the existing list
            bounding_box_list.append(bb)
            # editing mask
            mask[bb[1]:bb[3], bb[0]:bb[2]] = np.minimum(np.array(selected_digit), mask[bb[1]:bb[3], bb[0]:bb[2]]) # replacing mask with digit
            # adding annotation
            writer.addObject(digit_label, bb[0], bb[1], bb[2], bb[3])
    # save .xml annotation file
    writer.save(f"{FLAGS.data_dir}/Annotations/{example_name}.xml")
    # compose image
    mask = Image.fromarray(mask) # from numpy to Image
    random_back = random.choice(raw_backgrounds) # pick random background
    bg_img = Image.open(f"{bg_path}/{random_back}").resize([FLAGS.image_width, FLAGS.image_height]) # opne and resize
    random_front = random.choice(raw_foregrounds) # pick random foreground
    fg_img = Image.open(f"{fg_path}/{random_front}").resize([FLAGS.image_width, FLAGS.image_height])
    # random_color = tuple(np.random.randint(256, size= 3)) # pick random color for blending
    # random_color = Image.new(mode= 'RGB', size= [FLAGS.image_width, FLAGS.image_height])
    # fg_img = Image.blend(im1= fg_img, im2= random_color, alpha= 0.3)
    new_image = Image.composite(image1= bg_img, image2= fg_img, mask= mask)
    new_image.save(f"{FLAGS.data_dir}/JPEGImages/{example_name}.jpg", 'JPEG')

    return example_name
  

def main(_argv):

    # making project structure
    os.makedirs(f"{FLAGS.data_dir}/ImageSets", exist_ok= True)
    os.makedirs(f"{FLAGS.data_dir}/JPEGImages", exist_ok= True)
    os.makedirs(f"{FLAGS.data_dir}/Annotations", exist_ok= True)

    result = open(f"{FLAGS.data_dir}/ImageSets/{FLAGS.dataset_split}.txt", "a")
  
    # making samples
    for i in tqdm(range(FLAGS.n_samples)):
        image = generate_image(
        digit_path= f"{FLAGS.digit_dir}",
        bg_path= f"{FLAGS.bg_dir}",
        fg_path= f"{FLAGS.fg_dir}"
        )
        result.writelines([image, '\n'])
    logging.info(f"{FLAGS.n_samples} samples generated")
    result.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
