import numpy as np
from skimage.feature import canny
from scipy.ndimage.filters import sobel

MIN_CANNY_THRESHOLD = 10
MAX_CANNY_THRESHOLD = 50


def gradient_orientation(image):
    '''
    Calculate the gradient orientation for edge point in the image
    '''
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    gradient = np.arctan2(dy, dx) * 180 / np.pi
    return gradient


def build_r_table(img, ori):
    edges = canny(img, low_threshold=MIN_CANNY_THRESHOLD,
                  high_threshold=MAX_CANNY_THRESHOLD)
    gradient = gradient_orientation(edges)
    r_table = dict()
    x_idx, y_idx = np.nonzero(edges)
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        theta = gradient[x, y]
        if theta in r_table.keys():
            r_table[theta].append((ori[0]-x, ori[1]-y))
        else:
            r_table[theta] = [(ori[0]-x, ori[1]-y)]
    return r_table



