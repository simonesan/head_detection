"""
MODIFIED FROM https://github.com/AntonMu/TrainYourOwnYOLO
"""

from functools import reduce
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import re


def compose(*funcs):
    # print('----------------------compose-------------------------------')

    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


# TODO: check if needed
def letterbox_image(image, size):
    print('----------------------letterbox_image-------------------------------')

    # resize image with unchanged aspect ratio using padding
    image_width, image_height = image.size
    w, h = size
    scale = min(w / image_width, h / image_height)
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    image = image.resize((new_width, new_height), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - new_width) // 2, (h - new_height) // 2))
    return new_image


def rand(a=0, b=1):
    return np.random.uniform(a, b)


def distort_image(image, hue, sat, val):
    # get random hue/saturation/value value
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < 0.5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < 0.5 else 1 / rand(1, val)
    
    hsv_data = rgb_to_hsv(np.array(image) / 255.0)
    
    hsv_data[:, 0] += hue
    hsv_data[:, 1] *= sat
    hsv_data[:, 2] *= val
    
    # correct invalid values (<0 / >1)
    hsv_data[:, 0][hsv_data[:, 0] > 1] -= 1
    hsv_data[:, 0][hsv_data[:, 0] < 0] += 1
    hsv_data[hsv_data > 1] = 1
    hsv_data[hsv_data < 0] = 0
        
    return hsv_to_rgb(hsv_data)  # numpy array, 0 to 1


def make_grayscale(image_data):
    image_gray = np.dot(image_data, [0.299, 0.587, 0.114])
    return np.moveaxis(np.stack([image_gray, image_gray, image_gray]), 0, -1)


def invert_image(image_data):
    return 1 - image_data


def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def scale_box(box, image_width, image_height, new_width, new_height, dx, dy):
    box[:, [0, 2]] = box[:, [0, 2]] * new_width / image_width + dx
    box[:, [1, 3]] = box[:, [1, 3]] * new_height / image_height + dy
    return box


def flip_box(box, w):
    box[:, [0, 2]] = w - box[:, [2, 0]]
    return box


def correct_box_edges(box, w, h):
    # ensure box edges are within image
    box[:, 0:2][box[:, 0:2] < 0] = 0
    box[:, 2][box[:, 2] > w] = w
    box[:, 3][box[:, 3] > h] = h
    return box


def discard_invalid_boxes(box):
    # discard boxes where height or width < 1
    box_width = box[:, 2] - box[:, 0]
    box_height = box[:, 3] - box[:, 1]
    box = box[np.logical_and(box_width > 1, box_height > 1)]
    return box


def correct_boxes(box, max_boxes, image_width, image_height,
                  new_width, new_height, dx, dy,
                  input_width, input_height, flip):
    box_data = np.zeros((max_boxes, 5))
    
    if len(box) > 0:
        # shuffle order of boxes
        np.random.shuffle(box)
        
        # correct boxes based on allpied augmentation steps 
        box = scale_box(box, image_width, image_height, new_width, new_height, dx, dy)  
        if flip:
            box = flip_box(box, input_width)
        box = correct_box_edges(box, input_width, input_height)
        
        # discard invalid boxes
        box = discard_invalid_boxes(box)
        if len(box) > max_boxes:
            box = box[:max_boxes]
            
        box_data[: len(box)] = box
        
    return box_data


def get_image_and_boxes(line):
    # split at the first space that is followed by a number
    line = re.split("( \d)", line, maxsplit=1)
    if len(line) > 2:
        line = line[0], line[1] + line[2]

    # line[0] contains the filename
    image = Image.open(line[0])
    
    # extract boxes from rest of line
    line = line[1].split(" ")
    box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])

    return image, box


def resize_image(image, input_width, input_height, jitter):
    # get random aspect ratio and scale
    new_aspect_ratio = (input_width / input_height) * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(0.25, 2)
    
    # update width/height based on aspect ratio and scale
    if new_aspect_ratio < 1:
        new_height = int(scale * input_height)
        new_width = int(new_height * new_aspect_ratio)
    else:
        new_width = int(scale * input_width)
        new_height = int(new_width / new_aspect_ratio)
        
    image = image.resize((new_width, new_height), Image.BICUBIC)
    
    # position image randomly to fit new_width/height and input_width/height
    dx = int(rand(0, input_width - new_width))
    dy = int(rand(0, input_height - new_height))
    new_image = Image.new("RGB", (input_width, input_height), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    
    return new_image, new_width, new_height, dx, dy
    
    
# TODO: update parameter list
def get_random_data(annotation_line,
    input_shape,
    random=True,
    max_boxes=20,
    jitter=0.3,
    hue=0.1,
    sat=1.5,
    val=1.5,
    proc_img=True,
):
    # random preprocessing for real-time data augmentation

    image, box = get_image_and_boxes(annotation_line)
    image_width, image_height = image.size
    input_height, input_width = input_shape

    image, new_width, new_height, dx, dy = resize_image(image, input_width, input_height, jitter)

    flip = rand() < 0.5
    if flip:
        image = flip_image(image)       
        
    image_data = distort_image(image, hue, sat, val)
    
    if rand() < 0.2:
        image_data = make_grayscale(image_data)
        # print(image_data)
        
    if rand() < 0.1:
        image_data = invert_image(image_data)
        
    box_data = correct_boxes(box, max_boxes, image_width, image_height,
                             new_width, new_height, dx, dy,
                             input_width, input_height, flip)
        
    return image_data, box_data
