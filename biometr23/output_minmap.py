import numpy as np
import PIL.Image


def minutiae_map_to_RGB_Image(min_map):
    termins = min_map == 1
    bifurcs = min_map == 3

    outImg = np.zeros((min_map.shape[0], min_map.shape[1], 3), dtype=np.uint8)

    outImg[termins] = [255, 0, 0]
    outImg[bifurcs] = [0, 255, 0]

    return PIL.Image.fromarray(outImg, "RGB")


def overlay(base_img, over_img, alpha=0.5):
    return PIL.Image.blend(base_img, over_img, alpha)


def overlay3(base_img, mid_img, over_img, alpha=0.5):
    basemid = PIL.Image.blend(base_img, mid_img, alpha)
    return PIL.Image.blend(basemid, over_img, alpha)


def overlay_grey_bool_RGB(baseGrey, midBool, overRGB):
    """
    Overlay (blend/compose) 3 images of different modes,
    in order: greyscale PIL.Image, boolean ndarray, RGB PIL.Image

    :param baseGrey:
    :param midBool:
    :param overRGB:
    :return:
    """
    pass
