from do_hog import do_hog
from os import walk
from os.path import join
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure

# from 5.30, 2014
__author__ = 'Zhihua Liang'
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Zhihua Liang"
__email__ = "liangzhihua@gmail.com"
__status__ = "Development"


# loop over the sample folder and get samples and labels
def get_samples(sample_path, dim_x, dim_z, orientations, pixels_per_cell, cells_per_block, scan_window_size, print_image):
    # set to gray image
    plt.gray()
    #training samples and labels
    sample = []
    label = []
    lesions = []
    # hog size
    hog_x = dim_x / pixels_per_cell[0]
    hog_y = dim_z / pixels_per_cell[1]
    #loop over training image folder and get the histogram of gradient arrays
    print 'Getting samples and labels from files...'
    for root, dirs, files in walk(sample_path):
        for file_name in files:
            print file_name
            # get image path and lesion position
            file_path = join(root, file_name)
            lesion = bool(int(file_name.split('_lesion_')[-1].split('_')[0]))
            if lesion:
                lesion_x = int(file_name.split('_x_')[-1].split('_')[0])
                lesion_y = int(file_name.split('_z_')[-1].split('_')[0])
                # append the original position
                lesions.append([lesion_x, lesion_y])
                print 'lesion position:', lesion_x, lesion_y
                # calculate the lesion position in hog image for labels
                lesion_x = lesion_x / pixels_per_cell[0]
                lesion_y = lesion_y / pixels_per_cell[1]
            else:
                # if no lesion, they will be - 100
                lesion_x = -100
                lesion_y = -100
                lesions.append([lesion_x, lesion_y])
            # do hog
            if print_image:
                hog, hog_img = do_hog(file_path, dim_x, dim_z, orientations, pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block, visualise=print_image)
                hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 0.02))
            else:
                hog = do_hog(file_path, dim_x, dim_z, orientations, pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block, visualise=print_image)

            #get the strongest direction on each point
            hog = hog.reshape([hog_y, hog_x, orientations])
            hog = hog.argmax(axis=2)
            #scan window over the hog image, window size can varies
            for i in range(0, hog_y - scan_window_size[1]):
                for j in range(0, hog_x - scan_window_size[0]):
                    element = np.ndarray.flatten(hog[i:i + scan_window_size[1], j:j + scan_window_size[0]])
                    sample.append(element)
                    #if the window contains lesion
                    if i + scan_window_size[1] / 2 in range(lesion_y - 1, lesion_y + 2) \
                            and j + scan_window_size[0] / 2 in range(lesion_x - 1, lesion_x + 2):
                        print 'label 1 at:', (j + scan_window_size[1] / 2) * pixels_per_cell[1], \
                            (i + scan_window_size[0] / 2) * pixels_per_cell[0]
                        if print_image:
                            hog_img[j * pixels_per_cell[1]:(j + scan_window_size[1]) * pixels_per_cell[1],
                            i*pixels_per_cell[0]:(i + scan_window_size[0]) * pixels_per_cell[0]] += 0.02
                        label.append(1)
                    else:
                        label.append(0)
            if print_image:
                plt.imshow(hog_img)
                plt.savefig(join('figs', file_name.split('.img')[0]), format='png')
    return np.array(sample), np.array(label), lesions