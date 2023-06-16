import numpy as np
from biometr23.minutiae import minutiae_map

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
    minmap = minutiae_map(binImg)
    termins = minmap == 1
    bifurcs = minmap == 3

    # Remove spurious terminations at the bounds of the fingerprint.
    termins_hull = skimage.morphology.convex_hull_image(termins, True)
    termins_hull_eroded = skimage.morphology.binary_erosion(termins_hull,
                                                            footprint=np.full((11,11), True))
    termins_hull_bound = termins_hull ^ termins_hull_eroded
    plt.imshow(termins_hull_bound, cmap="binary")
    plt.colorbar()
    plt.show()
    termins[termins_hull_bound] = False

    #red_templ = np.full((minmap.shape[0], minmap.shape[1], 3), [255, 0, 0])
    #grn_templ = np.full((minmap.shape[0], minmap.shape[1], 3), [0, 255, 0])

    outImg = np.zeros((minmap.shape[0], minmap.shape[1], 3), dtype=np.uint8)

    outImg[termins] = [255, 0, 0]
    outImg[bifurcs] = [0, 255, 0]

    #skimage.io.imshow(minmap)
    #skimage.io.show()
    skimage.io.imsave("a.png", outImg)
    skimage.io.show()