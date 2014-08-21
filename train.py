__author__ = 'zhihua'

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


def train(classifier_file, training_path, dim_x, dim_z, orientations, pixels_per_cell, cells_per_block,
          scan_window_size, weight_values, distance_threshold=0.3,
          re_train=False, online_training=False, verbose=False):


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
                            clf.partial_fit(training_sample, training_label, classes=np.unique(training_label),
                                            sample_weight=sample_weight)
                            print 'training labels are', np.unique(training_label)
                        else:
                            clf.partial_fit(training_sample, training_label, sample_weight=sample_weight)
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
                        clf.partial_fit(training_sample, training_label, classes=np.unique(training_label),
                                    sample_weight=sample_weight)
                        print 'training labels are', np.unique(training_label)
                    else:
                        clf.partial_fit(training_sample, training_label, sample_weight=sample_weight)
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
    return clf