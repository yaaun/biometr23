import skimage.filters
import skimage.morphology

def binarize_below(img):
    """
    Binarize a greyscale image with some thresholding method.
    Mark the regions below the threshold that was found.

    Parameters
    ----------
    img : ndarray((rows, columns)) of int or float
        Greyscale image to binarize.

    Returns
    -------
    binImg : ndarray((rows, cols)) of bool
        Binary image with True pixels that had brightness below the threshold value found.
    """
    threshVal = skimage.filters.threshold_otsu(img)

    binImg = img < threshVal

    return binImg


def fill_holes(binImg):
    """
    Fill small holes in a binary image.

    Parameters
    ----------
    binImg

    Returns
    -------

    """
    filledImg = skimage.morphology.remove_small_holes()