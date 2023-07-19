import numpy as np
import scipy.ndimage
import skimage.morphology
import skimage.measure


# Taken and adapted to NumPy from Przemysław Pastuszka crossing_number.py (https://github.com/przemekpastuszka/biometrics)
def minutiae_at(pixels, y0, x0):
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    pxvals = np.array([pixels[y0 + dy, x0 + dx] for dy, dx in neighbours], dtype=np.int8)
    crossings2 = np.abs(np.diff(pxvals)).sum()

    if pixels[y0, x0]:
        if crossings2 == 2: # terminacja
            return 1
        if crossings2 == 6: # bifurkacja
            return 3
    return 0 # nic z tego

# Based on Przemysław Pastuszka crossing_number.py (https://github.com/przemekpastuszka/biometrics)
# Adapted to numpy array image representation
def minutiae_map(img):
    (max_y, max_x) = img.shape
    minutiae_locs = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i in range(1, max_y - 1):
        for j in range(1, max_x - 1):
            minutiae = minutiae_at(img, i, j)
            if minutiae is not None:
                minutiae_locs[i, j] = minutiae

    return minutiae_locs


def minutiae_map_filtered(img):
    """
    Map minutiae like :func:`minutiae_map` while filtering out
    boundary effects and fragmented fingerprint ridges.

    Parameters
    ----------
    img : ndarray((rows, columns))
        Binary image of skeletonized fingerprint.

    Returns
    -------
    minmap : ndarray((rows, columns)) of uint8
        Matrix of the same size as the input image with minutiae locations
        marked. 0 represents no feature, 1 represents a termination
        and 3 represents a bifurcation.

    """
    full3x3 = np.ones((3,3))

    # Settings for some size-specific morphology operations.
    # The values given below are very roughly estimated and definitely
    # not optimized.

    # The erosion matrix will be of this size.
    erosionDist = img.shape[1] // 40
    # The minimum line size, in pixels, will be this many.
    # Smaller (shorter) lines and objects will be ignored.
    minLineSize = img.shape[1] // 20

    # Assign numerical labels to distinct connected regions in the image.
    labelled, numObjs = scipy.ndimage.label(img, full3x3)

    # Remove small objects (short fragments of broken lines, in this case)
    # connectivity = 2 == full3x3 footprint, in this case
    noSmallObjsImg = skimage.morphology.remove_small_objects(labelled,
                                 min_size=minLineSize, connectivity=2)

    minmap = minutiae_map(noSmallObjsImg > 0) # run the minutiae mapping for surviving objects
    termins = minmap == 1
    bifurcs = minmap == 3

    # Remove terminations detected at the edge of the fingerprint.
    termins_hull = skimage.morphology.convex_hull_image(termins, True)
    termins_hull_eroded = skimage.morphology.binary_erosion(termins_hull,
                            footprint=np.full((erosionDist, erosionDist), True))
    termins_hull_bound = termins_hull ^ termins_hull_eroded
    minmap[termins_hull_bound] = 0 # or termins[termins_hull_bound] = False if termins is used further

    for minutiaeMap, minutiaeCode in zip((bifurcs,), (3,)):
        # Dilation of the found features to subsequently unify close lying objects.
        skimage.morphology.binary_closing(minutiaeMap, footprint=full3x3, out=minutiaeMap)

    return minmap


def calc_CoM(binMat):
    moments = skimage.measure.moments(binMat, order=1)
    xc = np.int32(np.round(moments[0, 1] / moments[0, 0]))
    yc = np.int32(round(moments[1, 0] / moments[0, 0]))

    return xc, yc

def bin_matrix_to_polar_coords(matrix, xc, yc):
    assert xc >= 0 and yc >= 0 and xc < matrix.shape[1] and yc < matrix.shape[0]

    coordsY, coordsX = np.nonzero(matrix)
    assert coordsX.shape == coordsY.shape and coordsX.ndim == 1 and coordsY.ndim == 1

    cX = coordsX - xc
    cY = coordsY - yc

    polarTable = np.zeros((2, len(coordsX)))

    polarTable[0] = np.sqrt(cX**2 + cY**2)
    polarTable[1] = np.arctan2(cY, cX) * 180 / np.pi

    return polarTable
