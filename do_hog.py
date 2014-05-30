# from 5.29, 2014
__author__ = 'Zhihua Liang'
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Zhihua Liang"
__email__ = "liangzhihua@gmail.com"
__status__ = "Development"

import numpy as np
from skimage.feature import hog


#read in image and return the histogram of gradient
def do_hog(file_path, dim_x, dim_z, orientations, pixels_per_cell, cells_per_block):
    """

    :param file_path:
    :param dim_x:
    :param dim_z:
    :param orientations:
    :param pixels_per_cell:
    :param cells_per_block:
    :return:
    """
    image = np.fromfile(file_path, dtype=np.float32)
    image = image.reshape([dim_z, dim_x])
    fd = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block)
    return fd

