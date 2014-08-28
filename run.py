#!/usr/bin/env python


import numpy as np
import os
from os import walk
from get_samples import get_hog_samples
from get_classifier import get_classifier
from get_location import get_location
from output import output
from os.path import join
from shutil import rmtree
from sklearn.externals import joblib as pickle
import progressbar
from train import train
from test import test

# from 5.29, 2014
__author__ = 'Zhihua Liang'
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Zhihua Liang"
__email__ = "liangzhihua@gmail.com"
__status__ = "Development"

#define the parameters
dim_x = 760
dim_y = 195
dim_z = 240
orientations = 9
target_size = 48
pixels_per_cell = (4, 4)
cells_per_block = (3, 3)  # not ready to change this value
weight_values = (1, 30)
distance_threshold = 0.01
scan_window_size = (target_size, target_size)  # on pixels
out_path = 'result'  # output directory
training_path = '/home/zhihua/work/object_detector/image/25_random_cleaned'
test_path = '/home/zhihua/work/object_detector/image/25_fix'
classifier_name = 'sgd'  # options are 'svm', 'sgd' for now
classifier_file = 'classifier/sgd.pkl'
re_train = False # only sgd get the retrain
online_training = True  # train on every single image when it is available.
verbose = True  # print debug message
use_distance_transform = True
#########################################################
# training
#########################################################
clf = train(classifier_file, training_path, classifier_name, dim_x, dim_z, orientations, pixels_per_cell,
            cells_per_block,
            scan_window_size, weight_values, use_distance_transform=use_distance_transform,
            distance_threshold=distance_threshold,
            re_train=re_train, online_training=online_training, verbose=verbose)
#########################################################
# test
#########################################################
#remove the previous output if there exist any
test(test_path, out_path, clf, dim_x, dim_z, target_size, orientations, pixels_per_cell, cells_per_block,
     scan_window_size, verbose)

# get the samples from test folder.
