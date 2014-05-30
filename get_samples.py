from do_hog import do_hog
from os import walk
from os.path import join
import numpy as np

# from 5.30, 2014
__author__ = 'Zhihua Liang'
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Zhihua Liang"
__email__ = "liangzhihua@gmail.com"
__status__ = "Development"


# loop over the sample folder and get samples and labels
def get_samples(sample_path, dim_x, dim_z, orientations, pixels_per_cell, cells_per_block, scan_window_size):
    #training samples and labels
    sample = []
    label = []
    lesions = []
    # hog size
    hog_x = dim_x/pixels_per_cell[0]
    hog_y = dim_z/pixels_per_cell[1]
    #loop over training image folder and get the histogram of gradient arrays
    print 'Getting samples and labels from files...'
    for root, dirs, files in walk(sample_path):
        for file_name in files:
            print file_name
            # get image path and lesion position
            file_path = join(root, file_name)
            lesion = bool(int(file_name.split('_')[19]))
            if lesion:
                lesion_x = int(file_name.split('_')[21])
                lesion_y = int(file_name.split('_')[25])
                # append the original position
                lesions.append([lesion_x, lesion_y])
                # calculate the lesion position in hog image for labels
                lesion_x = lesion_x/pixels_per_cell[0]
                lesion_y = lesion_y/pixels_per_cell[1]
            else:
                 # if no lesion, they will be - 100
                lesion_x = -100
                lesion_y = -100
                lesions.append([lesion_x, lesion_y])
            # do hog
            hog = do_hog(file_path, dim_x, dim_z, orientations, pixels_per_cell=pixels_per_cell,
                         cells_per_block=cells_per_block)
            hog = hog.reshape([hog_y, hog_x, orientations])
            #scan window over the hog image, window size can varies
            for i in range(0, hog_y-scan_window_size[1]):
                for j in range(0, hog_x-scan_window_size[0]):
                    element = np.ndarray.flatten(hog[i:i+scan_window_size[1], j:j+scan_window_size[0], :])
                    sample.append(element)
                    #if the window contains lesion
                    if i+scan_window_size[1]/2 in range(lesion_y-1, lesion_y+2) \
                            and j+scan_window_size[0]/2 in range(lesion_x-1, lesion_x+2):
                        label.append(1)
                    else:
                        label.append(0)
    return np.array(sample), np.array(label), lesions