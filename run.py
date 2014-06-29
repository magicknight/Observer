#!/usr/bin/env python


import numpy as np
from get_samples import get_hog_samples
from get_classifier import get_classifier
import os
import cPickle as pickle

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
target_size = 40
pixels_per_cell = (4, 4)
cells_per_block = (1, 1)  # not ready to change this value
scan_window_size = (target_size/pixels_per_cell[0], target_size/pixels_per_cell[1])  # on pixels
out_file = 'result.txt'
training_path = '/home/zhihua/work/HOG/image/training'
test_path = '/home/zhihua/work/HOG/image/temp'
svm_file = 'svm.pkl'
#########################################################
# training
#########################################################
if os.path.isfile(svm_file):
    #load SVM if there exist trained SVM file.
    with open(svm_file, 'rb') as fid:
        clf = pickle.load(fid)
else:
    #if no svm file exist, train it
    #training samples and labels
    print 'get training set'
    training_sample, training_label, dummy = get_hog_samples(training_path, dim_x, dim_z, orientations, pixels_per_cell,
                                                         cells_per_block, scan_window_size, print_image=False,
                                                         training=True)
    print 'Training set contains', len(training_label), 'samples'
    clf = get_classifier('svm', training_sample, training_label)
    with open(svm_file, 'wb') as fid:
        pickle.dump(clf, fid)

#########################################################
# test
#########################################################
# get the samples from test folder.
test_sample, test_label, lesion_positions = get_hog_samples(test_path, dim_x, dim_z, orientations, pixels_per_cell,
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
prediction_list = np.empty([])
with open(out_file, 'w') as fid:
    for i in range(n_samples):
        fid.write('=============== ' + 'Image ' + str(i) + ' =============== \n')
        fid.write(str(lesion_positions[i]))
        fid.write('--------------- \n')
        y, x = np.where(predict_label[i, :, :] == 1)
        predict_lesion_position = np.dstack((x*pixels_per_cell[0]+target_size/2.0, y*pixels_per_cell[1]+target_size/2.0))
        prediction_list = np.append(prediction_list, predict_lesion_position)
        fid.write('--------------- \n')
        fid.write(str(predict_lesion_position) + '\n')
with open('predict_positions.txt', 'w') as fid:
    pickle.dump(predict_lesion_position, fid)
