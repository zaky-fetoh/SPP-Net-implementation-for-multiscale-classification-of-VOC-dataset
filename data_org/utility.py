import xml.etree.ElementTree as ET
from data_org import *
import numpy as np
import cv2 as cv
import os

Decode = ['aeroplane', 'bicycle',
          'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
          'chair', 'cow', 'diningtable', 'dog',
          'horse', 'motorbike', 'person',
          'pottedplant', 'sheep', 'sofa',
          'train', 'tvmonitor', ]  # reverse Label Encoing
Encode = {x: i for i, x in enumerate(Decode)}  # label Encoding


# read images ids in the textfiles
def get_ids(labels_path):
    labels = list()
    with open(labels_path) as file:
        for line in file.readlines():
            labels.append(line[:-1])
    return labels


# parse the xml annoatation files to dictionary
def get_cls_bb(image_id):
    xmlfile_path = os.path.join(ANNOTAT_PATH, image_id + '.xml')
    root = ET.parse(xmlfile_path).getroot()
    image_data = dict()
    image_data['name'] = root.find('filename').text
    image_data['dims'] = (int(root.find('size/depth').text),
                          int(root.find('size/width').text),
                          int(root.find('size/height').text),)
    image_data['object'] = list()
    for obj in root.findall('object'):
        if int(obj.find('difficult').text):
            continue
        image_data['object'].append([
            obj.find('name').text,
            int(obj.find('bndbox/xmin').text),
            int(obj.find('bndbox/ymin').text),
            int(obj.find('bndbox/xmax').text),
            int(obj.find('bndbox/ymax').text),
        ])
    return image_data


# read aparticular image file with asecific id
def get_image(image_id):
    image_file_path = os.path.join(IMAGE_PATH, image_id + '.jpg')
    img_file = cv.imread(image_file_path)
    return img_file


def isotropically_scale(im, scale):
    r, c, _ = im.shape
    small_dim = min(r, c)
    nr, nc = r / small_dim,c/ small_dim
    nw, nh = int(nc * scale), int(nr * scale)
    return cv.resize(im,(nw,nh))


