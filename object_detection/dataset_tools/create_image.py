import os
import sys
# sys.path.append('/anaconda3/envs/whiteboard/lib/python3.7/site-packages')

import random
import numpy as np
import json

import matplotlib.pyplot as plt

# import tensorflow as tf
# flags = tf.app.flags
# flags.DEFINE_string('root_path', '', 'Path to output TFRecord')
# FLAGS = flags.FLAGS

ROOT_PATH='/Users/d070244/Code-Projects/whiteboard-training/'

FILEPATH_CIRCLES= os.path.join(ROOT_PATH, 'google_quickdraw_dataset/circle.ndjson')
FILEPATH_SQUARES= os.path.join(ROOT_PATH, 'google_quickdraw_dataset/square.ndjson')
FILEPATH_TRIANGLE=os.path.join(ROOT_PATH,  'google_quickdraw_dataset/triangle.ndjson')

DATASET_SAVE_FOLDER= ROOT_PATH + 'images/val/'

NB_TOTAL_OUTPUT_IMGS=5

NB_TOTAL_SQUARES = 120538
NB_TOTAL_CIRCLES = 118805
NB_TOTAL_TRIANGLES = 120499

NB_TOTAL_FIGURES = NB_TOTAL_SQUARES + NB_TOTAL_CIRCLES + NB_TOTAL_TRIANGLES

def select_figure():
    index = random.randint(0, NB_TOTAL_FIGURES)
    if index < NB_TOTAL_SQUARES:
        file = FILEPATH_SQUARES
        label = 'square'
    elif index < (NB_TOTAL_SQUARES + NB_TOTAL_CIRCLES):
        file = FILEPATH_CIRCLES
        index = index -  NB_TOTAL_SQUARES
        label = 'circle'
    else:
        file = FILEPATH_TRIANGLE
        index = index - NB_TOTAL_SQUARES - NB_TOTAL_CIRCLES
        label = 'triangle'

    with open(file) as f:
        for i, line in enumerate(f):
            if i == index:
                json_data = json.loads(line)
                break
    return json_data['drawing'], label


img_height = 1280
img_width = 1280



def create_image(nb_figures, name=None):
    plt.figure()
    plt.ylim(0, img_height)
    plt.xlim(0, img_width)
    plt.axis('off')


    if name is None:
        name = DATASET_SAVE_FOLDER + "example.jpeg"

    bbox_list = []
    for j in range(nb_figures):
        drawing, label = select_figure()

        min_x, min_y = img_width ,img_height
        max_x, max_y = 0 , 0

        if j%2==0:
            line_color = (random.uniform(0,1), random.uniform(0,1), random.uniform(0,1))
        line_width = random.uniform(1, 8)
        scale = random.uniform(0.25,2.5)

        max_offset_y = img_height - 255 * scale
        max_offset_x = img_width - 255 * scale

        offset = ( random.uniform(0, max_offset_x), random.uniform(0, max_offset_y))


        for line_segment in drawing:
            x = scale * np.array(line_segment[0], dtype=np.float) + offset[0]
            y = scale * np.array(line_segment[1], dtype=np.float) + offset[1]

            plt.plot(x, y, color=line_color, linewidth=line_width)

            min_x = min(x) if min(x) < min_x else min_x
            min_y = min(y) if min(y) < min_y else min_y
            max_x = max(x) if max(x) > max_x else max_x
            max_y = max(y) if max(y) > max_y else max_y

        max_x += line_width
        min_x -= line_width
        min_y -= line_width
        max_y += line_width
        bbox_list += [{
            'label': label,
            'xmin' : float(min_x)/float(img_width),
            'xmax': float(max_x)/float(img_width),
            'ymax' : 1.0 - float(min_y)/float(img_height),
            'ymin' : 1.0 - float(max_y)/float(img_height)

        }]

        # bbox = [[ min_x, min_x, max_x, max_x, min_x], [min_y, max_y, max_y, min_y, min_y] ]

        # plt.plot(bbox[0], bbox[1], color='tab:cyan')

    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.close()

    return name, bbox_list



def create_dataset(nb_images, nb_figures=3, root_path=ROOT_PATH, train=True):
    ''' Create jpeg images by randomly picking images from the dquickdraw dataset
    and pasting them in one canvas.
    Args:
        nb_images: An int the number of pictures the usar wants to generate
        nb_figures: int maximal number of figures per image
    Return:
        A list of dictionaries. Each dict has three attributes:
            'path': A string containing the absolute path to the image
            'bboxes': A list of dictionaries cotaining:
                'label': A string corresponding to the label
                'xmin' : float xmin/img_width
                'xmax': float xmax/img_width
                'ymin' : float ymin/img_height
                'ymax' : float ymax/img_height
    '''

    ## TODO:  fix global variables maybe passing them as a parameter
    global ROOT_PATH
    global FILEPATH_CIRCLES
    global FILEPATH_SQUARES
    global FILEPATH_TRIANGLE

    ROOT_PATH = root_path + os.path.sep

    FILEPATH_CIRCLES= os.path.join(ROOT_PATH, 'google_quickdraw_dataset/circle.ndjson')
    FILEPATH_SQUARES= os.path.join(ROOT_PATH, 'google_quickdraw_dataset/square.ndjson')
    FILEPATH_TRIANGLE=os.path.join(ROOT_PATH,  'google_quickdraw_dataset/triangle.ndjson')

    print(FILEPATH_TRIANGLE)
    print(FILEPATH_SQUARES)
    print(FILEPATH_CIRCLES)
    if train:
        DATASET_SAVE_FOLDER= os.path.join(ROOT_PATH, 'images/train/')
    else:
        DATASET_SAVE_FOLDER= os.path.join(ROOT_PATH, 'images/val/')

    list_imgs = []
    for i in range(nb_images):
        path, bboxes = create_image(nb_figures, DATASET_SAVE_FOLDER + str(i) + ".jpeg" )
        list_imgs += [{
            'path' : path,
            'bboxes': bboxes
        }]
    return list_imgs


images = create_dataset(3)
print(images)
