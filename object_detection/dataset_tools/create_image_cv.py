import os
import sys
# sys.path.append('/anaconda3/envs/whiteboard/lib/python3.7/site-packages')

import random
import numpy as np
import json

# import matplotlib.pyplot as plt
import cv2

# import tensorflow as tf
# flags = tf.app.flags
# flags.DEFINE_string('root_path', '', 'Path to output TFRecord')
# FLAGS = flags.FLAGS

ROOT_PATH='/Users/d070244/Code-Projects/WRA/whiteboard-training/google_quickdraw_dataset'

FILEPATH_CIRCLES= ROOT_PATH + 'google_quickdraw_dataset/circle.ndjson'
FILEPATH_SQUARES= ROOT_PATH + 'google_quickdraw_dataset/square.ndjson'
FILEPATH_TRIANGLE=ROOT_PATH +  'google_quickdraw_dataset/triangle.ndjson'

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


img_height = 1000
img_width = 1000

def select_figure2(data_folder='/Users/d070244/Code-Projects/WRA/whiteboard-training/google_quickdraw_dataset'):
    # data_folder='/Users/d070244/Code-Projects/WRA/whiteboard-training/google_quickdraw_dataset'
    files = os.listdir(data_folder)
    file = random.choice(files)
    label = file.split('.')[0]
    file = os.path.join( data_folder, file)
    num_lines = sum(1 for line in open(file))
    index = random.randint(0, (num_lines-1))
    # banned = [140, 99, 499, 237,181, 158  ]
    # print(index)
    # if label == 'arrow' and index in banned:
    #     index = random.randint(0, num_lines)
    # index =  140
    try:
        with open(file) as f:
            for i, line in enumerate(f):
                if i == index:
                    json_data = json.loads(line)
                    break
    except:
        print(label + " " + str(index))
        raise
        
    return json_data['drawing'], label


def create_image(max_nb_figures, name=None, data_folder='/workspace/23_Whiteboard_Recognition/whiteboard-training/google_quickdraw_dataset'):
    img = np.zeros((img_height,img_width,3), np.uint8)
    gray_offset = random.randint(230,255)# Not completely white
    img[:,:] = (gray_offset, gray_offset, gray_offset) # White image



    if name is None:
        name = DATASET_SAVE_FOLDER + "example.jpeg"

    bbox_list = []
    nb_figures = random.randint(2, max_nb_figures)
    for j in range(nb_figures):
        drawing, label = select_figure2(data_folder)

        min_x, min_y = img_width ,img_height
        max_x, max_y = 0 , 0

        if j%2==0:
            line_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        line_width = random.randint(1, 8)
        scale = random.uniform(0.01,1.5)

        max_offset_y = img_height - 255 * scale
        max_offset_x = img_width - 255 * scale

        offset = ( random.uniform(0, max_offset_x), random.uniform(0, max_offset_y))


        for line_segment in drawing:
            x = scale * np.array(line_segment[0], dtype=np.float) + offset[0]
            y = scale * np.array(line_segment[1], dtype=np.float) + offset[1]

            vertex = np.zeros((len(x), 2), np.int32)
            for i in range(len(x)):
                vertex[i] =  [int(x[i]), int(y[i])]

            vertex.reshape(-1,1,2)

            img = cv2.polylines(img, [vertex], False, line_color,line_width)
            # plt.plot(x, y, color=line_color, linewidth=line_width)

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
            'ymax' : float(max_y)/float(img_height),
            'ymin' : float(min_y)/float(img_height)

        }]

        # bbox = [[ min_x, min_x, max_x, max_x, min_x], [min_y, max_y, max_y, min_y, min_y] ]

        # plt.plot(bbox[0], bbox[1], color='tab:cyan')
    #Adding salt and pepper noise
    prob = 0.005
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                img[i][j] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    cv2.imwrite(name, img)
    # print(name)
    return name, bbox_list



def create_dataset(nb_images, nb_figures=4, root_path=ROOT_PATH, train=True, output_path=None):
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

    ROOT_PATH=root_path

    FILEPATH_CIRCLES= os.path.join(ROOT_PATH, 'google_quickdraw_dataset/circle.ndjson')
    FILEPATH_SQUARES= os.path.join(ROOT_PATH, 'google_quickdraw_dataset/square.ndjson')
    FILEPATH_TRIANGLE=os.path.join(ROOT_PATH,  'google_quickdraw_dataset/triangle.ndjson')

    if output_path is not None:
            if train:
                DATASET_SAVE_FOLDER= os.path.join(output_path, 'images/train/')
            else:
                DATASET_SAVE_FOLDER= os.path.join(output_path, 'images/val/')
    else:
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
        if i%100 == 0:
            print("Created {} images".format((i+1)))
    return list_imgs

if __name__ == '__main__':
    # create_image(5)
    create_dataset(10, output_path='/Users/d070244/Code-Projects/WRA/test_images')
