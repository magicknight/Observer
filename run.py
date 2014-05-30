#!/usr/bin/env python

import cPickle as pickle
import numpy as np
from get_samples import get_samples
from sklearn import svm
import os

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
svm_file = 'svm.pkl'
training_path = '/home/zhihua/work/HOG/image/training'
test_path = '/home/zhihua/work/HOG/image/small_test'

#load SVM if there exist trained SVM file.
if os.path.isfile(svm_file):
    with open(svm_file, 'rb') as fid:
        clf = pickle.load(fid)
#if no svm file exist, train it
else:
    #training samples and labels
    training_sample, training_label = get_samples(training_path, dim_x, dim_z, orientations, pixels_per_cell,
                                                  cells_per_block, scan_window_size)
    print 'Training set contains', len(training_label), 'samples'
    # training SVM and dump the trained svm to a binary file
    clf = svm.SVC()
    clf.fit(training_sample, training_label)
    with open(svm_file, 'wb') as fid:
        pickle.dump(clf, fid)

#get the samples from test folder.
test_sample, test_label = get_samples(test_path, dim_x, dim_z, orientations, pixels_per_cell,
                                      cells_per_block, scan_window_size)
predict_label = clf.predict(test_sample)
print 'Percentage error is:', np.mean(predict_label != test_label)