import numpy as np
from biometr23.minutiae import minutiae_map_filtered

import skimage.io
import skimage.morphology
import skimage.transform


if __name__ == "__main__":
    img = skimage.io.imread(
        #"bifurc_7x7.png",
        "skel_1_8.png",
        as_gray=True)
    binImg = img > 0
    minmap = minutiae_map_filtered(binImg)
    termins = minmap == 1
    bifurcs = minmap == 3

    imgMoments = skimage.measure.moments(img, 1)
    print_x0 = np.int32(np.round(imgMoments[0, 1] / imgMoments[0, 0]))
    print_y0 = np.int32(np.round(imgMoments[1, 0] / imgMoments[0, 0]))
    minmap[print_y0, print_x0] = 5

    minmapOut = np.zeros((minmap.shape[0], minmap.shape[1], 3), dtype=np.uint8)
    minmapOut[termins] = [255, 0, 0]
    minmapOut[bifurcs] = [0, 255, 0]
    minmapOut[minmap == 5] = [255, 255, 0] # centroid

    skimage.io.imsave("_out_pre_transform.png", minmapOut)

    minmapPolar = skimage.transform.warp_polar(minmap, center=(print_y0, print_x0), order=0)

    minmapPolarOut = np.zeros((minmapPolar.shape[0], minmapPolar.shape[1], 3), dtype=np.uint8)
    minmapPolarOut[minmapPolar == 1] = [255, 0, 0]
    minmapPolarOut[minmapPolar == 3] = [0, 255, 0]
    skimage.io.imsave("_out_polar.png", minmapPolarOut)

