#!/usr/bin/env python

import cPickle as pickle
import numpy as np
from get_samples import get_samples

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
pixels_per_cell = (4, 4)
cells_per_block = (1, 1) # not ready to change this value
scan_window_size = (9, 9) #on pixels

training_path = '/home/zhihua/work/HOG/image/training'
test_path = '/home/zhihua/work/HOG/image/test'

#training samples and labels
training_sample, training_label = get_samples(training_path, dim_x, dim_z, orientations, pixels_per_cell,
                                              cells_per_block, scan_window_size)
print training_label