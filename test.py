__author__ = 'zhihua'

import numpy as np
from os import walk
import os
from get_samples import get_hog_samples
from get_samples import distance_transform_samples
from get_location import get_location_random
from output import output
from os.path import join
from shutil import rmtree


def test(test_path, out_path, clf, dim_x, dim_z, target_size, orientations, pixels_per_cell, cells_per_block,
         scan_window_size, verbose, use_distance_transform=True):
    rmtree(out_path)
    os.makedirs(out_path)
    prediction_list = np.empty([])
    for root, dirs, files in walk(test_path):
        for file_name in files:
            print '==========================================================================='
            print file_name
            if use_distance_transform:
                test_sample, test_label, lesion_positions, locations = distance_transform_samples(join(root, file_name), dim_x,
                                                                                       dim_z,
                                                                                       orientations, pixels_per_cell,
                                                                                       cells_per_block, scan_window_size,
                                                                                       training=False, verbose=verbose)
            else:
                test_sample, test_label, lesion_positions, locations = get_hog_samples(join(root, file_name), dim_x, dim_z,
                                                                            orientations, pixels_per_cell,
                                                                            cells_per_block, scan_window_size,
                                                                            training=False, verbose=verbose)
            print 'Test set contains', len(test_label), 'samples'
            predict_label = clf.predict(test_sample)
            print 'Prediction-percentage-error is:', np.mean(predict_label != test_label)
            print np.where(np.array(test_label) == 1)
            print np.where(predict_label == 1)

            #go back to the original image axis
            #label_x = (dim_x - scan_window_size[0])/pixels_per_cell[0]+1
            #label_y = (dim_z - scan_window_size[1])/pixels_per_cell[1]+1
            #n_samples = len(lesion_positions)
            #print 'label number is', label_x*label_y
            #predict_label = predict_label.reshape([label_y, label_x])
            #y, x = np.where(predict_label[:, :] == 1)
            #predict_lesion_position = np.dstack((x*pixels_per_cell[0]+target_size/2,
            #                                     y*pixels_per_cell[1]+target_size/2))[0]
            locations = np.array(locations)
            predict_lesion_position = locations[np.where(predict_label == 1)]
            print 'candidate positions are:', predict_lesion_position
            # find the lesion location
            if predict_lesion_position.size != 0:
                #position, confidence = get_location_random(predict_lesion_position, target_size)
                position = get_location_random(predict_lesion_position)
                confidence = 4
            else:
                position = [-1, -1]
                confidence = 1
            #confidence = (confidence+1)/float(2)   # get to the range of LROC analysis
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