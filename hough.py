import imageio
import math
import numpy as np
from skimage.feature import canny
import matplotlib.pyplot as plt

MIN_CANNY_THRESHOLD = 10
MAX_CANNY_THRESHOLD = 50

def rgb2gray(img):
    return np.dot(img[...,:3], [0.229, 0.587, 0.114]).astype(np.uint8)

def hough_line(img, angle_step=2, r_step=2, threshold=5, lines_are_white=False):
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))

    radians = np.arange(0, 180, angle_step) * np.pi / 180.0
    cos_t = np.cos(radians)
    sin_t = np.sin(radians)

    num_radians = len(radians)
    accumulator = np.zeros([diag_len*2//r_step+1, num_radians+1])  # r 和 theta
    # 检测边缘
    # are_edges = canny(img, low_threshold=MIN_CANNY_THRESHOLD,
    #               high_threshold=MAX_CANNY_THRESHOLD)
    # plt.imshow(are_edges)
    # plt.show()
    # x_idx, y_idx = np.nonzero(are_edges)
    are_edges = img > threshold if lines_are_white else img < threshold
    x_idx, y_idx = np.nonzero(are_edges)

    x_dot = np.arange(0, 180, 1)
    for i in range(len(y_idx)):
        x = x_idx[i]
        y = y_idx[i]
        plt.plot(radians, x*np.cos(radians)+y*np.sin(radians))
        for j in range(num_radians):
            rho = (diag_len + int(round(x*cos_t[j]+y*sin_t[j])))//r_step
            try:
                accumulator[rho, j] += 1
            except:
                print(accumulator.shape,diag_len, rho, j)
            # if (rho,j*angle_step) in acc.keys():
            #     acc[(rho, j*angle_step)] += 1
            #     print(x, y, rho, j*angle_step)
            # else:
            #     acc[(rho, j * angle_step)] = 1
    plt.show()
    return accumulator


def show_image(img, acc, r_step=20, angle_step=5):
    plt.imshow(acc)
    plt.title('rho and theta')
    plt.show()

    plt.imshow(img, origin='default')
    width, height = img.shape
    plt.axis([0, height, 0, width])
    diag_len = int(round(np.sqrt(width * width + height * height)))
    x = np.arange(1, height, height//2-1)
    max_value = np.where(acc > np.max(acc)-1)
    for i in range(len(max_value[0])):
        max_rho_0 = max_value[0][i]
        max_rho = r_step * max_rho_0 - diag_len
        max_theta_0 = max_value[1][i]
        max_theta = angle_step * max_theta_0 * np.pi / 180.0
        # print(max_value, acc[max_rho_0, max_theta_0])
        plt.plot(x, (max_rho - x * np.cos(max_theta)) / np.sin(max_theta))
    plt.show()


if __name__ == '__main__':
    imgpath = 'images/dot.png'
    img = imageio.imread(imgpath)
    if img.ndim == 3:
        print("3 channel")
        img = rgb2gray(img)

    r_step = 1
    angle_step = 1
    acc = hough_line(img, r_step=r_step, angle_step=angle_step)
    show_image(img, acc, r_step=r_step, angle_step=angle_step)
