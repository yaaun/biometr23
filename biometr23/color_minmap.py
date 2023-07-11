import numpy as np
import PIL.Image


def minutiae_map_to_RGB_Image(min_map):
    termins = min_map == 1
    bifurcs = min_map == 3

    outImg = np.zeros((min_map.shape[0], min_map.shape[1], 3), dtype=np.uint8)

    outImg[termins] = [255, 0, 0]
    outImg[bifurcs] = [0, 255, 0]

    return PIL.Image.fromarray(outImg, "RGB")
