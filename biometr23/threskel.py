from collections import namedtuple
import PIL.Image
import scipy.optimize
from PIL import Image
import numpy as np
from skimage import morphology
from skimage.util import img_as_bool


FitParamsGauss = namedtuple("FitParamsGauss", "mu sigma a")

def gauss(x, mu, sigma, a):
    """Formula for gaussian fitting"""
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))




def fit_gauss_hist256(histCounts, pad_length=64, pad_val=0, paramsHint=(None, None, None)):
    assert len(histCounts) == 256

    y = np.pad(np.asarray(histCounts), pad_length, constant_values=pad_val)
    x = np.arange(0 - pad_length, 256 + pad_length)

    #expected = (170, 50, 10000)  # the starting parameters for the fit
    #expected = (30, 20, img.width * img.height / 20)
    #expected = (15, 10, img.width * img.height / 20)
    expected = (
        paramsHint[0] if paramsHint[0] is not None else 15,
        paramsHint[1] if paramsHint[1] is not None else 10,
        paramsHint[2] if paramsHint[2] is not None else 10000
    )

    params, covMatrix = scipy.optimize.curve_fit(
        f=gauss,
        xdata=x,
        ydata=y,
        p0=expected
    )

    return FitParamsGauss(*params)


def create_gauss_threshold_LUT(gaussParams, meanShift=20):
    range256 = np.arange(0, 256) # 255 + 1 z powodu reguł [zamkniętego; otwartego) przedziału w Pythonie
    boolMask = range256 <= gaussParams[0] + meanShift
    lut = boolMask * 255
    return lut


def autothreshold(img, thresh_hint=None, thresh_shift=None):
    # If there is problem with gaussian fitting uncomment three lines below
    #  to see the histogram fit and choose better starting parameters in expected

    # plt.plot(np.arange(0, 256), y, label='Test data')
    # plt.plot(np.arange(0, 256), gauss(np.arange(0, 256), *params), label='Fitted data')
    # plt.show()

    # The threshold for binarization is set as the center of the peak minus
    # three time standard deviation. Depending on the quality of scan it may be
    # required to choose smaller multiplier than 3.

    #image_binary = img.point(lambda p: 0 if p > params[0] - 3 * params[1] else 255)

    if thresh_hint is not None:
        gaussParams = fit_gauss_hist256(img.histogram(), paramsHint=(thresh_hint, None, None))
    else:
        gaussParams = fit_gauss_hist256(img.histogram())

    if thresh_shift is not None:
        image_binary = img.point(create_gauss_threshold_LUT(gaussParams, thresh_shift))
    else:
        image_binary = img.point(create_gauss_threshold_LUT(gaussParams))

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
