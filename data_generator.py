from PIL import Image
import numpy as np
import xmlify
import os
from pascal_voc_writer import Writer
import data_generation_config

#getting parameters
number_of_samples = data_generation_config._number_of_samples_
image_size = data_generation_config._image_size_
generation_path = os.path.join(os.getcwd(), data_generation_config._generation_path_)
source_path = os.path.join(os.getcwd(), data_generation_config._source_path_)
raw_digit_path = os.path.join(source_path, 'raw_digits')
raw_backgrounds_path = os.path.join(source_path, 'raw_backgrounds')

#make directory for generated images
os.makedirs(generation_path, exist_ok= True)

#making samples
for sample in range(number_of_samples):

    #select random digit
    #with transformation
    random_digit = np.random.randint(10)
    digit_wh = np.random.randint(30, 120)
    selected_digit = Image.open(f'{raw_digit_path}/{random_digit}.jpg').convert('L')
    selected_digit = selected_digit.resize((digit_wh, digit_wh))
    selected_digit = selected_digit.rotate(np.random.randint(-20, 20), fillcolor= 'white')

    #locate bounding box
    
    x_min = np.random.randint(int(image_size[0] * 0.05), int(image_size[0] * 0.7))
    y_min = np.random.randint(int(image_size[1] * 0.05), int(image_size[1] * 0.7))
    x_max = x_min + digit_wh
    y_max = y_min + digit_wh

    #make mask
    mask = Image.new(mode= 'L', size= image_size, color= 'white')
    mask = np.array(mask)
    mask[y_min:y_max, x_min:x_max] = np.array(selected_digit)
    mask = Image.fromarray(mask)

    #foreground & background
    fg_color = tuple(np.random.randint(256, size= 3))
    fg_img = Image.new(mode= 'RGB', size= image_size, color= fg_color)
    random_back = np.random.randint(1, 6)
    bg_img = Image.open(f'{raw_backgrounds_path}/{random_back}.jpg').resize(image_size)

    #composing & saving
    image_path = f'{generation_path}/{sample}.jpg'
    new_image = Image.composite(image1= bg_img, image2= fg_img, mask= mask)
    new_image.save(image_path, 'JPEG')

    #annotations
    writer = Writer(image_path, image_size[0], image_size[1])
    writer.addObject('digit', x_min, y_min, x_max, y_max)
    writer.save(f'{generation_path}/{sample}.xml')