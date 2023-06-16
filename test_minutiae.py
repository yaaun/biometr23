import numpy as np
from biometr23.minutiae import minutiae_map_filtered

import matplotlib as mpl
import matplotlib.pyplot as plt

import skimage.io
import skimage.morphology

if __name__ == "__main__":
    img = skimage.io.imread(
        #"bifurc_7x7.png",
        "skel_1_8.png",
        as_gray=True)
    binImg = img > 0
    minmap = minutiae_map_filtered(binImg)
    termins = minmap == 1
    bifurcs = minmap == 3

    #red_templ = np.full((minmap.shape[0], minmap.shape[1], 3), [255, 0, 0])
    #grn_templ = np.full((minmap.shape[0], minmap.shape[1], 3), [0, 255, 0])

    outImg = np.zeros((minmap.shape[0], minmap.shape[1], 3), dtype=np.uint8)

    outImg[termins] = [255, 0, 0]
    outImg[bifurcs] = [0, 255, 0]

    #skimage.io.imshow(minmap)
    #skimage.io.show()
    skimage.io.imsave("_out.png", outImg)
    #skimage.io.show()