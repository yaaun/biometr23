import PIL.Image
from scipy.optimize import curve_fit
from PIL import Image
import numpy as np
from skimage import morphology
from skimage.util import img_as_bool


def gauss(x, mu, sigma, a):
    """Formula for gaussian fitting"""
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def autothreshold(img):
    PAD_LENGTH = 64
    PAD_VAL = 0
    #y = np.array(img.histogram())
    y = np.pad(np.array(img.histogram()), PAD_LENGTH, constant_values=PAD_VAL)
    x = np.arange(0 - PAD_LENGTH, 256 + PAD_LENGTH)

    #expected = (170, 50, 10000)  # the starting parameters for the fit
    #expected = (30, 20, img.width * img.height / 20)
    #expected = (15, 10, img.width * img.height / 20)
    expected = (15, 10, 10000)

    result = curve_fit(
        f=gauss,
        xdata=x,
        ydata=y,
        p0=expected)

    params, cov = result

    # If there is problem with gaussian fitting uncomment three lines below
    #  to see the histogram fit and choose better starting parameters in expected

    # plt.plot(np.arange(0, 256), y, label='Test data')
    # plt.plot(np.arange(0, 256), gauss(np.arange(0, 256), *params), label='Fitted data')
    # plt.show()

    # The threshold for binarization is set as the center of the peak minus
    # three time standard deviation. Depending on the quality of scan it may be
    # required to choose smaller multiplier than 3.

    #image_binary = img.point(lambda p: 0 if p > params[0] - 3 * params[1] else 255)
    image_binary = img.point(lambda p: 0 if p > params[0] + 20 else 255)

    return img_as_bool(image_binary)


def skeletonize_clean(img_binary):
    # get rid of pores
    no_holes = morphology.remove_small_holes(img_as_bool(img_binary), 15)
    skeleton = morphology.thin(no_holes)
    # get rid of small artefacts
    no_small_obj = morphology.remove_small_objects(skeleton, 5, connectivity=2)

    return no_small_obj


def process(img):
    return skeletonize_clean(autothreshold(img))
