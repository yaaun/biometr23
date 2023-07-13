import numpy as np
import PIL.Image
import skimage.util


def minutiae_map_to_RGB_Image(min_map):
    termins = min_map == 1
    bifurcs = min_map == 3

    outImg = np.zeros((min_map.shape[0], min_map.shape[1], 3), dtype=np.uint8)

    outImg[termins] = [255, 0, 0]
    outImg[bifurcs] = [0, 255, 0]

    return PIL.Image.fromarray(outImg, "RGB")


def minutiae_map_to_RGBA_Image(min_map):
    termins = min_map == 1
    bifurcs = min_map == 3

    outImg = np.zeros((min_map.shape[0], min_map.shape[1], 4), dtype=np.uint8)

    outImg[termins] = [255, 0, 0, 255]
    outImg[bifurcs] = [0, 255, 0, 255]

    return PIL.Image.fromarray(outImg, "RGBA")


def overlay(base_img, over_img, alpha=0.5):
    return PIL.Image.blend(base_img, over_img, alpha)


def overlay3(base_img, mid_img, over_img, alpha=0.5):
    basemid = PIL.Image.blend(base_img, mid_img, alpha)
    return PIL.Image.blend(basemid, over_img, alpha)


def overlay_grey_bool_RGB(baseGrey, midBool, overRGB):
    """
    Overlay (blend/compose) 3 images of different modes,
    in order: greyscale PIL.Image, boolean ndarray, RGB PIL.Image

    Parameters
    ----------
    baseGrey : PIL.Image

    """
    assert baseGrey.mode == "L"
    assert midBool.dtype == np.bool_
    assert overRGB.mode == "RGB"
    midImg = PIL.Image.fromarray(midBool * np.uint8(255), "L")
    midImgTranspMask = PIL.Image.fromarray((~midBool) * np.uint8(255), "L")


def overlay_grey_bool_RGBA(baseGrey, midBool, overRGBA, alphaMid=0.5):
    """
    Overlay (blend/compose) 3 images of different modes,
    in order: greyscale PIL.Image, boolean ndarray, RGB PIL.Image

    Parameters
    ----------
    baseGrey : PIL.Image

    """
    assert baseGrey.mode == "L"
    assert midBool.dtype == np.bool_
    assert overRGBA.mode == "RGBA"

    if 0.0 <= alphaMid <= 1:
        alphaMid *= alphaMid
    elif 0 <= alphaMid <= 255:
        pass
    else:
        raise ValueError("alphaMid must be an integer from 0 to 255 or a float from 0.0 to 1.0.")

    midImg = PIL.Image.fromarray(midBool * np.uint8(255), "L")
    midImgMask = PIL.Image.fromarray((midBool) * np.uint8(alphaMid), "L")

    base_mid = PIL.Image.composite(midImg, baseGrey, midImgMask) # WARNING: PIL.Image.composite compositing base is actually the 2nd image!

    base_mid_over = PIL.Image.alpha_composite(base_mid.convert("RGBA"), overRGBA)

    return base_mid_over.convert("RGB")


def overlay_grey_RGBA(baseGrey, overRGBA):
    return PIL.Image.alpha_composite(baseGrey.convert("RGBA"), overRGBA).convert("RGB")