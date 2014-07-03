from do_hog import do_hog
from os import walk
from os.path import join
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
import random

# from 5.30, 2014
__author__ = 'Zhihua Liang'
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Zhihua Liang"
__email__ = "liangzhihua@gmail.com"
__status__ = "Development"


def block_shaped(arr, n_rows, n_cols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//n_rows, n_rows, -1, n_cols)
               .swapaxes(1, 2)
               .reshape(-1, n_rows, n_cols))


# loop over the sample folder and get hog samples and labels
def get_hog_samples(file_path, dim_x, dim_z, orientations, pixels_per_cell, cells_per_block, scan_window_size,
                print_image, training=False):
    # set to gray image
    #training samples and labels
    sample = []
    label = []
    lesions = []
    # positive and negative samples for training only
    positive_samples = []
    negative_samples = []
    # hog size
    hog_x = dim_x / pixels_per_cell[0]
    hog_y = dim_z / pixels_per_cell[1]
    #loop over training image folder and get the histogram of gradient arrays
    print 'Getting samples and labels from files...'
    file_name = file_path.split('/')[-1]
    print file_name
    # get image path and lesion position
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
        plt.gray()
        hog, hog_img = do_hog(file_path, dim_x, dim_z, orientations, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block, visualise=print_image)
        hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 0.02))
    else:
        hog = do_hog(file_path, dim_x, dim_z, orientations, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block, visualise=print_image)

    #get the strongest direction on each point, try not to do this
    hog = hog.reshape([hog_y, hog_x, orientations])
    #hog = hog.argmax(axis=2)
    #hog = hog/float(orientations-1) #scale the features between [0,1]
    # #scan window over the hog image, window size can varies
    for i in range(0, hog_y - scan_window_size[1]):
        for j in range(0, hog_x - scan_window_size[0]):
            element = np.ndarray.flatten(hog[i:i + scan_window_size[1], j:j + scan_window_size[0], :])

            #if the window contains lesion
            if i + scan_window_size[1] / 2 in range(lesion_y - 1, lesion_y + 2) \
                    and j + scan_window_size[0] / 2 in range(lesion_x - 1, lesion_x + 2):
                print 'label 1 at:', (j + scan_window_size[0] / 2) * pixels_per_cell[0], \
                    (i + scan_window_size[1] / 2) * pixels_per_cell[1]
                if print_image:
                    hog_img[j * pixels_per_cell[0]:(j + scan_window_size[0]) * pixels_per_cell[0],
                        i*pixels_per_cell[1]:(i + scan_window_size[1]) * pixels_per_cell[1]] += 0.02
                    plt.imshow(hog_img[i*pixels_per_cell[0]:(i + scan_window_size[1])*pixels_per_cell[0],
                                j*pixels_per_cell[1]:(j + scan_window_size[0])*pixels_per_cell[1]])
                    plt.savefig(join('figs', file_name.split('.img')[0]+str(i)+str(j)), format='png', dpi=200)
                if training:
                    positive_samples.append(element)
                else:
                    sample.append(element)
                    label.append(1)
            # get rid of windows that contains part of the lesion
            elif i in range(int(lesion_y - 1.5*scan_window_size[1]), int(lesion_y + 0.5*scan_window_size[1])) \
                    and j in range(int(lesion_x - 1.5*scan_window_size[0]), int(lesion_x + 0.5*scan_window_size[0])):
                # for training, get rid of those contain part of the lesions, try not to confuse the classifier
                if training:
                    continue
                else:
                    sample.append(element)
                    label.append(0)
            # window that contains no lesion
            else:
                if training:
                    negative_samples.append(element)
                else:
                    sample.append(element)
                    label.append(0)
    if print_image:
        plt.imshow(hog_img)
        plt.savefig(join('figs', file_name.split('.img')[0]), format='png', dpi=200)
    if training:
        print 'selecting negative samples ', len(negative_samples)/10, 'totally.'
        print 'positive samples is', len(positive_samples), 'totally.'
        negative_samples = random.sample(negative_samples, len(negative_samples)/10)
        sample = negative_samples + positive_samples
        label = [0]*len(negative_samples)+[1]*len(positive_samples)
    return sample, label, lesions


# loop over the sample folder and get samples with a smaller window
def get_pre_train_samples(sample_path, dim_x, dim_y, scan_window_size):
    # set to gray image
    #training samples and labels
    sample = []
    n_sub_image = [dim_x/scan_window_size[0], dim_y/scan_window_size[1]]
    #loop over training image folder and get the histogram of gradient arrays
    print 'Getting samples and labels from files...'
    for root, dirs, files in walk(sample_path):
        for file_name in files:
            print file_name
            # get image path and lesion position
            file_path = join(root, file_name)
            image = np.fromfile(file_path, dtype=np.float32).reshape([dim_y, dim_x])
            print image.max()
            sub_images = block_shaped(image, scan_window_size[1], scan_window_size[0])
            [sample.append(y.ravel()) for y in sub_images]
    return sample

