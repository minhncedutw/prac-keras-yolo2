#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Description
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "CONG-MINH NGUYEN"
__copyright__ = "Copyright (C) 2019, prac-keras-yolo2"
__credits__ = ["CONG-MINH NGUYEN"]
__license__ = "GPL"
__version__ = "1.0.1"
__date__ = "2019-05-05"
__maintainer__ = "CONG-MINH NGUYEN"
__email__ = "minhnc.edu.tw@gmail.com"
__status__ = "Development"  # ["Prototype", "Development", or "Production"]
# Project Style: https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6
# Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting

#==============================================================================
# Imported Modules
#==============================================================================
import argparse
import os.path
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"

import json
import numpy as np
import cv2

from keras.models import load_model

from preprocessing import parse_annotation
from frontend_re import YOLO
from utils import draw_boxes

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================

#==============================================================================
# Main function
#==============================================================================
def _main_(args):
    print('Hello World! This is {:s}'.format(args.desc))

    '''**************************************************************
    I. Set parameters
    '''
    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    '''**************************************************************
    II. Prepare the data
    '''
    # 1: Parse dataset
    train_imgs, train_labels = parse_annotation(
        ann_dir =config['train']['train_annot_folder'],
        img_dir =config['train']['train_image_folder'],
        labels  =config['model']['labels']
    )
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(
            ann_dir=config['valid']['valid_annot_folder'],
            img_dir=config['valid']['valid_image_folder'],
            labels=config['model']['labels']
        )
    else:
        train_valid_split = int(0.8 * len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels.keys()

    '''**************************************************************
    III. Create the model
    '''
    # 1: Build the model architecture.
    yolo = YOLO(
        backend             =config['model']['backend'],
        input_size          =config['model']['input_size'],
        labels              =config['model']['labels'],
        max_box_per_image   =config['model']['max_box_per_image'],
        anchors             =config['model']['anchors']
    )

    # 2: Load some weights into the model.
    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])

    '''**************************************************************
    IV. Kick off the training
    '''
    yolo.train(
        train_imgs          =train_imgs,
        valid_imgs          =valid_imgs,
        train_times         =config['train']['train_times'],
        valid_times         =config['valid']['valid_times'],
        nb_epoch            =config['train']['nb_epochs'],
        learning_rate       =config['train']['learning_rate'],
        batch_size          =config['train']['batch_size'],
        warmup_epochs       =config['train']['warmup_epochs'],
        object_scale        =config['train']['object_scale'],
        no_object_scale     =config['train']['no_object_scale'],
        coord_scale         =config['train']['coord_scale'],
        class_scale         =config['train']['class_scale'],
        saved_weights_name  =config['train']['saved_weights_name'],
        debug               =config['train']['debug']
    )

    '''**************************************************************
    V. Test & Evaluate
    '''
    # 1: Set test parameters.
    input_path = args.input
    output_path = args.output
    weights_path = args.weights

    # 2: Obtain test image paths.
    image_paths = []
    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]
    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'jpeg'])]

    # 3: Load model.
    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])
    yolo.load_weights(weights_path)

    # 4: Make predictions.
    for image_path in image_paths:
        image = cv2.imread(image_path)
        print(image_path)

        # predict the bounding boxes
        boxes = yolo.predict(image)

        # draw bounding boxes on the image using labels
        image = draw_boxes(image, boxes, config['model']['labels'])

        # display predicted results
        cv2.imshow("A", np.uint8(image))
        cv2.waitKey(0)

        # write the image with bounding boxes to file
        cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))




if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='YOLO2!!!')
    argparser.add_argument('-d', '--desc', help='description of the program', default='Train and validate YOLOv2 model on Raccoon dataset')
    argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')
    argparser.add_argument('-i', '--input',
                           default='/home/minhnc-lab/WORKSPACES/AI/Samples/keras-yolo2/images/raccoon/',
                           help='path to an image, a directory of images, a video, or webcam')
    argparser.add_argument('-o', '--output',
                           default='/home/minhnc-lab/WORKSPACES/AI/Samples/keras-yolo2/output/',
                           help='path to output directory')
    argparser.add_argument('-w', '--weights',
                           default='./tiny_yolo_raccoon.h5',
                           help='path to the weights of trained model')

    args = argparser.parse_args()
    _main_(args)
