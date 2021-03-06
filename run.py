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
scan_window_size = (target_size, target_size)  # on pixels
out_path = 'result'  # output directory
training_path = '/home/zhihua/work/object_detector/image/25_50_75_0_1'
test_path = '/home/zhihua/work/object_detector/image/25_50_75_2'
classifier_name = 'sgd'  # options are 'svm', 'sgd' for now
classifier_file = 'classifier/sgd.pkl'
re_train = False # only sgd get the retrain
online_training = True  # train on every single image when it is available.
verbose = False  # print debug message
#########################################################
# training
#########################################################
# get progress bar for display progress
bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
# get number of files in training directory
number_of_total_files = sum([len(files) for r, d, files in os.walk(training_path)])
number_of_total_files_over_20 = number_of_total_files/100 + 1
file_count = 0

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
                                                                         training=True, verbose=verbose)
                if online_training:
                    n_positive = np.count_nonzero(training_label)
                    sample_weight = [weight_values[0]]*(len(training_label) - n_positive) + [weight_values[1]]*n_positive
                    if file_count == 0:
                        clf.partial_fit(training_sample, training_label, classes=np.unique(training_label))
                                       # sample_weight=sample_weight)
                        print 'training labels are', np.unique(training_label)
                    else:
                        clf.partial_fit(training_sample, training_label)#, sample_weight=sample_weight)
                else:
                    total_training_sample = total_training_sample + training_sample
                    total_training_label = total_training_label + training_label
                file_count += 1
                #print 're-Training set contains', len(total_training_label), 'samples'
                if file_count/number_of_total_files_over_20 == float(file_count)/float(number_of_total_files_over_20):
                    bar.update(file_count/number_of_total_files_over_20)
        if not online_training:
            clf.partial_fit(total_training_sample, total_training_label)  # WARNING: Only SGD get the
                                                                      # online learning feature.
        pickle.dump(clf, classifier_file)
# if no svm exist, create it and train
else:
    #if no svm file exist, train it
    clf = get_classifier(classifier_name)
    #training samples and labels
    print 'Get training set on', training_path
    print 'Training on progress.... \n\n\n\n'
    for root, dirs, files in walk(training_path):
        for file_name in files:
            training_sample, training_label, dummy = get_hog_samples(join(root, file_name), dim_x, dim_z,
                                                                     orientations, pixels_per_cell,
                                                                     cells_per_block, scan_window_size,
                                                                     training=True, verbose=verbose)
            if online_training:
                n_positive = np.count_nonzero(training_label)
                sample_weight = [weight_values[0]]*(len(training_label) - n_positive) + [weight_values[1]]*n_positive
                if file_count == 0:
                    clf.partial_fit(training_sample, training_label, classes=np.unique(training_label))
                                    #sample_weight=sample_weight)
                    print 'training labels are', np.unique(training_label)
                else:
                    clf.partial_fit(training_sample, training_label)#, sample_weight=sample_weight)
            else:
                total_training_sample = total_training_sample + training_sample
                total_training_label = total_training_label + training_label
            file_count += 1
            if file_count/number_of_total_files_over_20 == float(file_count)/float(number_of_total_files_over_20):
                bar.update(file_count/number_of_total_files_over_20)
    if not online_training:
        print '\n Training set contains', len(total_training_label), 'samples'
        print total_training_sample[0].shape
        clf.fit(total_training_sample, total_training_label)
    pickle.dump(clf, classifier_file)
bar.finish()
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
            print '==========================================================================='
            print file_name
            test_sample, test_label, lesion_positions = get_hog_samples(join(root, file_name), dim_x, dim_z,
                                                                        orientations, pixels_per_cell,
                                                                        cells_per_block, scan_window_size,
                                                                        training=False, verbose=verbose)
            print 'Test set contains', len(test_label), 'samples'
            predict_label = clf.predict(test_sample)
            print 'Prediction-percentage-error is:', np.mean(predict_label != test_label)
            print np.where(np.array(test_label) == 1)
            print np.where(predict_label == 1)

            #go back to the original image axis
            label_x = (dim_x - scan_window_size[0])/pixels_per_cell[0]+1
            label_y = (dim_z - scan_window_size[1])/pixels_per_cell[1]+1
            #n_samples = len(lesion_positions)
            print 'label number is', label_x*label_y
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
            confidence = (confidence+1)/float(2)   # get to the range of LROC analysis
            print 'predicted location is', position, 'with confidence', confidence
            lesion = int(file_name.split('lesion_')[-1].split('_')[0]) > 0
            truth_x = int(file_name.split('x_')[-1].split('_')[0])
            truth_y = int(file_name.split('z_')[-1].split('_')[0])
            if lesion:
                print 'truth position is    ', [truth_x, truth_y]
            else:
                print 'truth position is    ', [-1, -1]
            # get the density value and projection number as output file name:
            projection_number = file_name.split('tvp_')[-1].split('_')[0]
            output_file_name = 'TP_' + projection_number + '.txt'
            #open out put file
            with open(join(out_path, output_file_name), 'a') as fid:
                output(file_name, position, confidence, fid)