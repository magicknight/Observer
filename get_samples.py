from do_hog import do_hog
from os import walk
from os.path import join
import numpy as np
import random
import cv2
from skimage.feature import peak_local_max

# from 5.30, 2014
__author__ = 'Zhihua Liang'
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Zhihua Liang"
__email__ = "liangzhihua@gmail.com"
__status__ = "Development"

positive_threshold = 10

def block_shaped(arr, n_rows, n_cols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // n_rows, n_rows, -1, n_cols)
            .swapaxes(1, 2)
            .reshape(-1, n_rows, n_cols))


# loop over the sample folder and get hog samples and labels
def get_hog_samples(file_path, dim_x, dim_y, orientations, pixels_per_cell, cells_per_block, scan_window_size,
                    training=False, verbose=False):
    # training samples and labels
    """

    :param file_path:
    :param dim_x:
    :param dim_y:
    :param orientations:
    :param pixels_per_cell:
    :param cells_per_block:
    :param scan_window_size:
    :param training:
    :param verbose:
    :return:
    """
    sample = []
    label = []
    lesions = []
    positions = []
    # positive and negative samples for training only
    positive_samples = []
    negative_samples = []
    # number of block on x and y
    n_cells_x = dim_x / pixels_per_cell[0]
    n_cells_y = dim_y / pixels_per_cell[1]
    n_blocks_x = (n_cells_x - cells_per_block[0]) + 1
    n_blocks_y = (n_cells_y - cells_per_block[1]) + 1
    n_blocks_scan_window = scan_window_size[0] / pixels_per_cell[0] - cells_per_block[0] + 1
    if verbose:
        print 'scan window contains', n_blocks_scan_window, 'blocks.'

    # loop over training image folder and get the histogram of gradient arrays
    file_name = file_path.split('/')[-1]
    # get image path and lesion position
    lesion = bool(int(file_name.split('_lesion_')[-1].split('_')[0]))
    if lesion:
        lesion_x = int(file_name.split('_x_')[-1].split('_')[0])
        lesion_y = int(file_name.split('_z_')[-1].split('_')[0])
        # append the original position
        lesions.append([lesion_x, lesion_y])
    else:
        # if no lesion, they will be - 100
        lesion_x = -100
        lesion_y = -100
        lesions.append([lesion_x, lesion_y])
    # do HOG
    hog = do_hog(file_path=file_path, dim_x=dim_x, dim_z=dim_y, orientations=orientations,
                 pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                 normalise=True)
    hog = hog.reshape([n_blocks_y, n_blocks_x, cells_per_block[1], cells_per_block[0], orientations])
    if verbose:
        print 'hog has shape', hog.shape
    #scan window over the image, window size can varies
    for j in range(0, n_blocks_y - n_blocks_scan_window + 1):
        for i in range(0, n_blocks_x - n_blocks_scan_window + 1):
            j_in_pixel = j * pixels_per_cell[1]
            i_in_pixel = i * pixels_per_cell[0]
            element = np.ndarray.flatten(hog[j:j + n_blocks_scan_window, i:i + n_blocks_scan_window, :])
            #if verbose:
            #    print 'i in pixel:', i_in_pixel
            #    print 'j in pixel:', j_in_pixel

            #if the window contains lesion
            if j_in_pixel + scan_window_size[1] / 2 in range(lesion_y - positive_threshold+1, lesion_y + positive_threshold) \
                    and i_in_pixel + scan_window_size[0] / 2 in range(lesion_x - positive_threshold+1, lesion_x + positive_threshold):
                # print 'label 1 in', i_in_pixel + scan_window_size[0]/2, j_in_pixel + scan_window_size[1]/2
                if training:
                    positive_samples.append(element)
                else:
                    sample.append(element)
                    label.append(1)
                    positions.append([i_in_pixel, j_in_pixel])
            # get rid of windows that contains part of the lesion
            elif j_in_pixel in range(int(lesion_y - 1.5 * scan_window_size[1]),
                                     int(lesion_y + 0.5 * scan_window_size[1])) \
                    and i_in_pixel in range(int(lesion_x - 1.5 * scan_window_size[0]),
                                            int(lesion_x + 0.5 * scan_window_size[0])):
                # for training, get rid of those contain part of the lesions, try not to confuse the classifier
                if training:
                    continue
                else:
                    sample.append(element)
                    label.append(0)
                    positions.append([i_in_pixel, j_in_pixel])
            # window that contains no lesion
            else:
                if training:
                    negative_samples.append(element)
                else:
                    sample.append(element)
                    label.append(0)
                    positions.append([i_in_pixel, j_in_pixel])
    if training:
        # get random data set from the whole data set.
        #negative_samples = random.sample(negative_samples, len(negative_samples)/10)
        sample = negative_samples + positive_samples
        label = [0] * len(negative_samples) + [1] * len(positive_samples)
    if verbose:
        print file_name
        print 'sample contains', len(sample), 'elements'
        if training:
            print 'negative sample has', len(negative_samples), ', position sample has', len(positive_samples)
    return sample, label, lesions, positions


# loop over the sample folder and get samples with a smaller window
def get_pre_train_samples(sample_path, dim_x, dim_y, scan_window_size):
    # set to gray image
    # training samples and labels
    sample = []
    n_sub_image = [dim_x / scan_window_size[0], dim_y / scan_window_size[1]]
    # loop over training image folder and get the histogram of gradient arrays
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


# ################################### get distance transform based samples ####################
# loop over the sample folder and get hog samples and labels
def distance_transform_samples(file_path, dim_x, dim_y, orientations, pixels_per_cell, cells_per_block,
                               scan_window_size, distance_threshold=0.3,
                               training=False, verbose=False):
    """
    This function get distance transform based samples. Do hog and get 30% of the maximum bright region as samples

    :param file_path:
    :param dim_x:
    :param dim_y:
    :param orientations:
    :param pixels_per_cell:
    :param cells_per_block:
    :param scan_window_size:
    :param training:
    :param verbose:
    """
    sample = []
    label = []
    lesions = []
    positions = []
    # positive and negative samples for training only
    positive_samples = []
    negative_samples = []
    # number of block on x and y
    n_cells_x = dim_x / pixels_per_cell[0]
    n_cells_y = dim_y / pixels_per_cell[1]
    n_blocks_x = (n_cells_x - cells_per_block[0]) + 1
    n_blocks_y = (n_cells_y - cells_per_block[1]) + 1
    n_blocks_scan_window = scan_window_size[0] / pixels_per_cell[0] - cells_per_block[0] + 1
    if verbose:
        print 'scan window contains', n_blocks_scan_window, 'blocks.'

    # loop over training image folder and get the histogram of gradient arrays
    file_name = file_path.split('/')[-1]
    # get image path and lesion position
    lesion = bool(int(file_name.split('_lesion_')[-1].split('_')[0]))
    if lesion:
        lesion_x = int(file_name.split('_x_')[-1].split('_')[0])
        lesion_y = int(file_name.split('_z_')[-1].split('_')[0])
        # append the original position
        lesions.append([lesion_x, lesion_y])
    else:
        # if no lesion, they will be - 100
        lesion_x = -100
        lesion_y = -100
        lesions.append([lesion_x, lesion_y])
    # do HOG
    hog = do_hog(file_path=file_path, dim_x=dim_x, dim_z=dim_y, orientations=orientations,
                 pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                 normalise=True)
    hog = hog.reshape([n_blocks_y, n_blocks_x, cells_per_block[1], cells_per_block[0], orientations])
    if verbose:
        print 'hog has shape', hog.shape

    # do distance transform and find out local maximum
    image = np.fromfile(file_path, dtype=np.float32)
    image = image.reshape([dim_y, dim_x])
    gray = np.array(image * 255, dtype=np.uint8)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, distance_threshold * dist_transform.max(), 255, 0)
    local_maxi = peak_local_max(dist_transform, indices=False, footprint=np.ones((4, 4)))
    local_maxi[sure_fg == 0] = 0
    locations = np.argwhere(local_maxi == 1)

    #scan window over the image, window size can varies
    for j_in_pixel, i_in_pixel in locations:
        j = j_in_pixel / pixels_per_cell[1]
        i = i_in_pixel / pixels_per_cell[0]
        if i < n_blocks_scan_window / 2 or j < n_blocks_scan_window / 2 or j > n_blocks_y - n_blocks_scan_window / 2 or i > n_blocks_x - n_blocks_scan_window / 2:
            continue
        element = np.ndarray.flatten(hog[j - n_blocks_scan_window / 2:j + n_blocks_scan_window / 2,
                                     i - n_blocks_scan_window / 2:i + n_blocks_scan_window / 2, :])
        #element = np.ndarray.flatten(hog[j:j+n_blocks_scan_window, i:i+n_blocks_scan_window, :])
        #if verbose:
           # print 'i in pixel and j in pixel:', i_in_pixel, j_in_pixel

        #if the window contains lesion
        if j_in_pixel in range(lesion_y - positive_threshold+1, lesion_y + positive_threshold) \
                and i_in_pixel in range(lesion_x - positive_threshold+1, lesion_x + positive_threshold):
            # print 'label 1 in', i_in_pixel + scan_window_size[0]/2, j_in_pixel + scan_window_size[1]/2
            if training:
                positive_samples.append(element)
            else:
                sample.append(element)
                label.append(1)
                positions.append([i_in_pixel, j_in_pixel])
        # get rid of windows that contains part of the lesion
        elif j_in_pixel in range(int(lesion_y - 0.5 * scan_window_size[1]), int(lesion_y + 0.5 * scan_window_size[1])) \
                and i_in_pixel in range(int(lesion_x - 0.5 * scan_window_size[0]),
                                        int(lesion_x + 0.5 * scan_window_size[0])):
            # for training, get rid of those contain part of the lesions, try not to confuse the classifier
            if training:
                continue
            else:
                sample.append(element)
                label.append(0)
                positions.append([i_in_pixel, j_in_pixel])
                # window that contains no lesion
        else:
            if training:
                negative_samples.append(element)
            else:
                sample.append(element)
                label.append(0)
                positions.append([i_in_pixel, j_in_pixel])
    if training:
        # get random data set from the whole data set.
        #negative_samples = random.sample(negative_samples, len(negative_samples)/10)
        sample = negative_samples + positive_samples
        label = [0] * len(negative_samples) + [1] * len(positive_samples)
    if verbose:
        print file_name
        print 'sample contains', len(sample), 'elements'
        if training:
            print 'negative sample has', len(negative_samples), ', position sample has', len(positive_samples)
        if np.count_nonzero(label) == 0 and lesion_x != -100:
            print '!!!!!!!!             no positive position        !!!!!!!!!!!!!'
    return sample, label, lesions, positions
