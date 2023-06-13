import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import scipy.ndimage
import skimage.io
import skimage.morphology

def main():
    #inimg = np.uint8(255 * skimage.io.imread("BW_swirl.png", as_gray=True))
    inImg = skimage.io.imread("BW_swirl.png", as_gray=True)
    skelImg = skimage.morphology.skeletonize(inImg)
    skimage.io.imshow(skelImg)
    structElem = np.full((3,3), 1)
    labelled, numObj = scipy.ndimage.label(skelImg, structElem)
    skimage.io.show()


if __name__ == "__main__":
    main()
