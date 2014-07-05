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
target_size = 52
pixels_per_cell = (4, 4)
cells_per_block = (1, 1)  # not ready to change this value
scan_window_size = (target_size/pixels_per_cell[0], target_size/pixels_per_cell[1])  # on pixels
out_path = 'result'  # output directory
training_path = '/home/zhihua/work/object_detector/image/training'
test_path = '/home/zhihua/work/object_detector/image/pa60_all'
classifier_name = 'sgd'  # options are 'svm', 'sgd' for now
classifier_file = 'SGD.pkl'
re_train = False # only sgd get the retrain
#########################################################
# training
#########################################################
total_training_sample = []
total_training_label = []
if os.path.isfile(classifier_file):
    #load SVM if there exist trained SVM file.
    clf = pickle.load(classifier_file)
    # continue train the model with new data
    if re_train:
        print 'get re-training set'
        for root, dirs, files in walk(training_path):
            for file_name in files:
                training_sample, training_label, dummy = get_hog_samples(join(root, file_name), dim_x, dim_z,
                                                                         orientations, pixels_per_cell,
                                                                         cells_per_block, scan_window_size,
                                                                         print_image=False,
                                                                         training=True)
                total_training_sample = total_training_sample + training_sample
                total_training_label = total_training_label + training_label
                print 're-Training set contains', len(total_training_label), 'samples'
        clf.partial_fit(total_training_sample, total_training_label)  # WARNING: Only SGD get the
                                                                      # online learning feature.
        pickle.dump(clf, classifier_file)
# if no svm exist, create it and train
else:
    #if no svm file exist, train it
    clf = get_classifier(classifier_name)
    #training samples and labels
    print 'get training set'
    for root, dirs, files in walk(training_path):
        for file_name in files:
            training_sample, training_label, dummy = get_hog_samples(join(root, file_name), dim_x, dim_z,
                                                                     orientations, pixels_per_cell,
                                                                     cells_per_block, scan_window_size,
                                                                     print_image=False,
                                                                     training=True)
            total_training_sample = total_training_sample + training_sample
            total_training_label = total_training_label + training_label
    print 'Training set contains', len(total_training_label), 'samples'
    clf.fit(total_training_sample, total_training_label)
    pickle.dump(clf, classifier_file)

#########################################################
# test
#########################################################
#remove the previous output if there exist any
rmtree(out_path)
os.makedirs(out_path)
# get the samples from test folder.
prediction_list = np.empty([])
for root, dirs, files in walk(test_path):
        for file_name in files:
            test_sample, test_label, lesion_positions = get_hog_samples(join(root, file_name), dim_x, dim_z,
                                                                        orientations, pixels_per_cell,
                                                                        cells_per_block, scan_window_size,
                                                                        print_image=False,
                                                                        training=False)
            print 'Test set contains', len(test_label), 'samples'
            predict_label = clf.predict(test_sample)
            print 'Prediction-percentage-error is:', np.mean(predict_label != test_label)
            print np.where(np.array(test_label) == 1)
            print np.where(predict_label == 1)

            #go back to the original image axis
            label_x = dim_x/pixels_per_cell[0] - scan_window_size[0]
            label_y = dim_z/pixels_per_cell[1] - scan_window_size[1]
            #n_samples = len(lesion_positions)
            predict_label = predict_label.reshape([label_y, label_x])
            y, x = np.where(predict_label[:, :] == 1)
            predict_lesion_position = np.dstack((x*pixels_per_cell[0]+target_size/2,
                                                 y*pixels_per_cell[1]+target_size/2))[0]
            print 'candidate positions are:', predict_lesion_position
            # find the lesion location
            if predict_lesion_position.size != 0:
                position, confidence = get_location(predict_lesion_position, target_size)
            else:
                position = [-1, -1]
                confidence = 1
            print 'predicted location is', position, 'with confidence', confidence
            # get the density value and projection number as output file name:
            density = file_name.split('Pd')[-1].split('_')[0]
            projection_number = file_name.split('tp_')[-1].split('_')[0]
            output_file_name = 'PD_' + density + '_TP_' + projection_number + '.txt'
            #open out put file
            with open(join(out_path, output_file_name), 'a') as fid:
                output(file_name, position, confidence, fid)