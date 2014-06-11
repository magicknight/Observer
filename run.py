#!/usr/bin/env python


import numpy as np
from get_samples import get_samples
from get_classifier import get_classifier

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
pixels_per_cell = (10, 10)
cells_per_block = (1, 1)  # not ready to change this value
scan_window_size = (40/pixels_per_cell[0], 40/pixels_per_cell[1])  # on pixels
out_file = 'result.txt'
training_path = '/home/zhihua/work/HOG/image/training'
test_path = '/home/zhihua/work/HOG/image/small_test'

#########################################################
# training
#########################################################
#training samples and labels
print 'get training set'
training_sample, training_label, dummy = get_samples(training_path, dim_x, dim_z, orientations, pixels_per_cell,
                                                         cells_per_block, scan_window_size, print_image=False)
print 'Training set contains', len(training_label), 'samples'
clf = get_classifier('svm', training_sample, training_label)

#########################################################
# test
#########################################################
# get the samples from test folder.
test_sample, test_label, lesion_positions = get_samples(test_path, dim_x, dim_z, orientations, pixels_per_cell,
                                                        cells_per_block, scan_window_size, print_image=False)
print 'Test set contains', len(test_label), 'samples'
predict_label = clf.predict(test_sample)
print 'Prediction-percentage-error is:', np.mean(predict_label != test_label)
print np.where(test_label == 1)
print np.where(predict_label == 1)
#go back to the original image axis
label_x = dim_x/pixels_per_cell[0] - scan_window_size[0]
label_y = dim_z/pixels_per_cell[1] - scan_window_size[1]
n_samples = len(lesion_positions)
predict_label = predict_label.reshape([n_samples, label_y, label_x])
with open(out_file, 'w') as fid:
    for i in range(n_samples):
        fid.write('=============== ' + 'Image ' + str(i) + ' =============== \n')
        y, x = np.where(predict_label[:, :, i] == 1)
        predict_lesion_position = np.dstack((x*pixels_per_cell[0], y*pixels_per_cell[1]))
        fid.write(str(lesion_positions[i]) + '\n')
        fid.write('--------------- \n')
        fid.write(str(predict_lesion_position) + '\n')
    fid.write(str(test_label))
    fid.write(str(predict_label))
